import os
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.layers.core import Lambda
from keras.layers import ReLU
from keras.layers import LeakyReLU
from keras.layers import Conv2D
from keras.layers import UpSampling2D
from keras.layers import BatchNormalization
from keras.engine.topology import Layer
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model

class split0(Layer):
	def call(self, inputs):
		return inputs[:,:,:,:3]
	def compute_output_shape(self, input_shape):
		return tuple([input_shape[0], input_shape[1], input_shape[2], 3])
		
class split1(Layer):
	def call(self, inputs):
		return inputs[:,:,:,3:6]
	def compute_output_shape(self, input_shape):
		return tuple([input_shape[0], input_shape[1], input_shape[2], 3])
	
class split2(Layer):
	def call(self, inputs):
		return inputs[:,:,:,6:9]
	def compute_output_shape(self, input_shape):
		return tuple([input_shape[0], input_shape[1], input_shape[2], 3])
		
class Mask(Layer):
	def call(self, inputs):
		output = inputs[0] * inputs[1]
		return output
	def compute_output_shape(self, input_shape):
		return input_shape[0]
		
class change_value(Layer):
	def call(self, inputs):
		normalization = K.mean(inputs, axis = [1, 2], keepdims = True) #均值
		normalization = K.repeat_elements(normalization, inputs.shape[1], axis=1)
		normalization = K.repeat_elements(normalization, inputs.shape[2], axis=2)
		return normalization
		
	def compute_output_shape(self, input_shape):
		return input_shape
		
class generate_mask(Layer):
	def __init__(self, output_dim, knsize, **kwargs):
		self.output_dim = output_dim
		self.kernel_size = [knsize, knsize]
		super(generate_mask, self).__init__(**kwargs)

	def call(self, inputs):
		mask_output = K.conv2d(
			inputs, K.ones((self.kernel_size[0], self.kernel_size[0], inputs.shape[3], self.output_dim)),
			strides=[2, 2],
			padding="same"
		)
		mask_output = K.cast(K.greater(mask_output, 0), "float32")
		return mask_output
	def compute_output_shape(self, input_shape):
		return tuple([input_shape[0], input_shape[1] // 2, input_shape[2] // 2, self.output_dim])
	
class concat_layer(Layer):
	def call(self, inputs):
		return K.concatenate([inputs[0], inputs[1]], axis = 3)
	def compute_output_shape(self, input_shape):
		return tuple([input_shape[0][0], input_shape[0][1], input_shape[0][2], input_shape[0][3] + input_shape[1][3]])	
'''调用形式: LossLayers(初始化列表)(9个layers的list)'''
class LossLayer(Layer):
	def __init__(self, **kwargs):
		self.is_placeholder = True
		super(LossLayer, self).__init__(**kwargs)
		
	def loss_hole(self, mask, gt, predic):
		return self.l1((1 - mask) * gt, (1 - mask) * predic)
		
	def loss_valid(self, mask, gt, predic):
		return self.l1(mask * gt, mask * predic)
	#predic, gt 在vgg16跑下来pool1、pool2、pool3的
	def loss_perceptual(self, vgg_predic, vgg_gt):
		return self.l1(vgg_predic, vgg_gt)
	#loss_styleout(predict)
	def loss_style(self, vgg_predic, vgg_gt):
		vgg_predic_t = vgg_predic
		vgg_predic_t = tf.transpose(vgg_predic_t, perm = [0,3,2,1])
		vgg_predic = tf.transpose(vgg_predic, perm = [0,3,1,2])
		vgg_gt_t = vgg_gt
		vgg_gt_t = tf.transpose(vgg_gt_t, perm = [0,3,2,1])
		vgg_gt = tf.transpose(vgg_gt, perm = [0,3,1,2])
		return self.l1(tf.matmul(vgg_predic_t, vgg_predic), tf.matmul(vgg_gt_t, vgg_gt))
	
	#平滑度计算，先对遮挡区域适当扩大（因为需要关注和非遮挡区域的平滑度）
	def loss_variation(self, mask, predic):
		kernel = K.ones((3, 3, mask.shape[3], mask.shape[3]))
		dilated_holes = K.conv2d(1 - mask, kernel, data_format = "channels_last", padding = "same")
		dilated_holes = K.cast(K.greater(dilated_holes, 0), "float32")
		ret = dilated_holes * predic
		a0 = self.l1(ret[:,1:,:,:], ret[:,:-1,:,:])
		a1 = self.l1(ret[:,:,1:,:], ret[:,:,:-1,:])
		return a0 + a1
	def l1(self, A, B):
		tmp = K.abs(A - B)
		return K.mean(tmp)

	#inputs为list, 应该传入predic, gt, mask, (pool0, pool1, pool2)*(predic, gt, comp) 共9个参数 传了pool0,再传pool1
	def call(self, inputs):
		loss = 0.0
		loss += 6.0 * self.loss_hole(inputs[2], inputs[1], inputs[0])
		loss += self.loss_valid(inputs[2], inputs[1], inputs[0])
		for i in range(3, 11, 3):
			loss += 0.05 * self.loss_perceptual(inputs[i], inputs[i + 1])
			loss += (1 / K.cast((inputs[i].shape[3] * inputs[i].shape[3]), "float32")) * 180.0 * self.loss_style(inputs[i], inputs[i + 1])
			loss += 0.05 * self.loss_perceptual(inputs[i + 2], inputs[i + 1])
		loss += 0.1 * self.loss_variation(inputs[2], inputs[0] * (1 - inputs[2]) + inputs[1] * inputs[2])
		loss = K.cast(loss, "float64")
		self.add_loss(loss, inputs = inputs)
		
		return inputs[0]
		
'''伪UNet 比pconv原论文少了两层(encode 和 decode各少一层) ，
使用BN，Adam， loss函数有一点不同，实现起来更方便（希望结果不要太差，T^T）'''

'''拆分'''
inputs = keras.Input(shape = [256, 256, 9])
gt = split0()(inputs)
mask0 = split1()(inputs)
x0_ma = split2()(inputs)
tmp = change_value()(mask0)
x0_val = Lambda(lambda x: x[0] / x[1])([x0_ma, tmp])


'''conv1 512*512*3 to 256*256*64'''
x1 = Conv2D(filters = 64,
			kernel_size = [7, 7],
			strides = [2, 2],
			padding = "same",
			activation = "relu")(x0_val)
mask1 = generate_mask(64, 7)(mask0)
x1_ma = Mask()([x1, mask1])
tmp = change_value()(mask1)
x1_val = Lambda(lambda x: x[0] / x[1])([x1_ma, tmp])

'''conv2 256*256*64 to 128*128*128'''	
x2 = Conv2D(filters = 128,
			kernel_size = [5, 5],
			strides = [2, 2],
			padding = "same",
			activation = None)(x1_val)
x2 = BatchNormalization(axis = 3, name = "Batch2")(x2)
x2 = ReLU()(x2)
mask2 = generate_mask(128, 5)(mask1)
x2_ma = Mask()([x2, mask2])
tmp = change_value()(mask2)
x2_val = Lambda(lambda x: x[0] / x[1])([x2_ma, tmp])

'''conv3 128*128*128 to 64*64*256'''
x3 = Conv2D(filters = 256,
			kernel_size = [5, 5],
			strides = [2, 2],
			padding = "same",
			activation = None)(x2_val)
x3 = BatchNormalization(axis = 3, name = "Batch3")(x3)
x3 = ReLU()(x3)
mask3 = generate_mask(256, 5)(mask2)
x3_ma = Mask()([x3, mask3])
tmp = change_value()(mask3)
x3_val = Lambda(lambda x: x[0] / x[1])([x3_ma, tmp])

'''conv4 64*64*256 to 32*32*512'''
x4 = Conv2D(filters = 512,
			kernel_size = [3, 3],
			strides = [2, 2],
			padding = "same",
			activation = None)(x3_val)
x4 = BatchNormalization(axis = 3, name = "Batch4")(x4)
x4 = ReLU()(x4)
mask4 = generate_mask(512, 3)(mask3)
x4_ma = Mask()([x4, mask4])
tmp = change_value()(mask4)
x4_val = Lambda(lambda x: x[0] / x[1])([x4_ma, tmp])

'''conv5 32*32*512 to 16*16*512'''
x5 = Conv2D(filters = 512,
			kernel_size = [3, 3],
			strides = [2, 2],
			padding = "same",
			activation = None)(x4_val)
x5 = BatchNormalization(axis = 3, name = "Batch5")(x5)
x5 = ReLU()(x5)
mask5 = generate_mask(512, 3)(mask4)
x5_ma = Mask()([x5, mask5])
tmp = change_value()(mask5)
x5_val = Lambda(lambda x: x[0] / x[1])([x5_ma, tmp])

'''conv6 16*16*512 to 8*8*512'''
x6 = Conv2D(filters = 512,
			kernel_size = [3, 3],
			strides = [2, 2],
			padding = "same",
			activation = None)(x5_val)
x6 = BatchNormalization(axis = 3, name = "Batch6")(x6)
x6 = ReLU()(x6)
mask6 = generate_mask(512, 3)(mask5)
x6_ma = Mask()([x6, mask6])
tmp = change_value()(mask6)
x6_val = Lambda(lambda x: x[0] / x[1])([x6_ma, tmp])

'''conv7 8*8*512 to 4*4*512'''
x7 = Conv2D(filters = 512,
			kernel_size = [3, 3],
			strides = [2, 2],
			padding = "same",
			activation = None)(x6_val)
x7 = BatchNormalization(axis = 3, name = "Batch7")(x7)
x7 = ReLU()(x7)
mask7 = generate_mask(512, 3)(mask6)
x7_ma = Mask()([x7, mask7])
tmp = change_value()(mask7)
x7_val = Lambda(lambda x: x[0] / x[1])([x7_ma, tmp])

'''
downsample

upsample
'''

'''conv8   x7_up and merge with x6_val 4*4*512 to 8*8*512 to 8*8*(512+512) to 8*8*512'''
x7_up = UpSampling2D(size = [2, 2], interpolation="nearest")(x7_val)
x8 = concat_layer()([x7_up, x6_val])
x8 = Conv2D(filters = 512,
			kernel_size = [3, 3],
			strides = [1, 1],
			padding = "same",
			activation = None)(x8)
x8 = BatchNormalization(axis = 3)(x8)
x8 = LeakyReLU(alpha = 0.2)(x8)
			
'''conv9   x8_up and merge with x5_val 8*8*512 to 16*16*512 to 16*16*(512+512) to 16*16*512'''
x8_up = UpSampling2D(size = [2, 2], interpolation="nearest")(x8)
x9 = concat_layer()([x8_up, x5_val])
x9 = Conv2D(filters = 512,
			kernel_size = [3, 3],
			strides = [1, 1],
			padding = "same",
			activation = None)(x9)
x9 = BatchNormalization(axis = 3)(x9)
x9 = LeakyReLU(alpha = 0.2)(x9)
			
'''conv10   x9_up and merge with x4_val 16*16*512 to 32*32*512 to 32*32*(512+512) to 32*32*512'''
x9_up = UpSampling2D(size = [2, 2], interpolation="nearest")(x9)
x10 = concat_layer()([x9_up, x4_val])
x10 = Conv2D(filters = 512,
			kernel_size = [3, 3],
			strides = [1, 1],
			padding = "same",
			activation = None)(x10)
x10 = BatchNormalization(axis = 3)(x10)
x10 = LeakyReLU(alpha = 0.2)(x10)
			
'''conv11   x10_up and merge with x3_val 32*32*512 to 64*64*512 to 64*64*(512+256) to 64*64*256'''
x10_up = UpSampling2D(size = [2, 2], interpolation="nearest")(x10)
x11 = concat_layer()([x10_up, x3_val])
x11 = Conv2D(filters = 256,
			kernel_size = [3, 3],
			strides = [1, 1],
			padding = "same",
			activation = None)(x11)
x11 = BatchNormalization(axis = 3)(x11)
x11 = LeakyReLU(alpha = 0.2)(x11)

'''conv12   x11_up and merge with x2_val 64*64*256 to 128*128*256 to 128*128*(256+128) to 128*128*128'''
x11_up = UpSampling2D(size = [2, 2], interpolation="nearest")(x11)
x12 = concat_layer()([x11_up, x2_val])
x12 = Conv2D(filters = 128,
			kernel_size = [3, 3],
			strides = [1, 1],
			padding = "same",
			activation = None)(x12)
x12 = BatchNormalization(axis = 3)(x12)
x12 = LeakyReLU(alpha = 0.2)(x12)
			
'''conv13   x12_up and merge with x1_val 128*128*128 to 256*256*128 to 256*256*(128+64) to 256*256*64'''
x12_up = UpSampling2D(size = [2, 2], interpolation="nearest")(x12)
x13 = concat_layer()([x12_up, x1_val])
x13 = Conv2D(filters = 64,
			kernel_size = [3, 3],
			strides = [1, 1],
			padding = "same",
			activation = None)(x13)
x13 = BatchNormalization(axis = 3)(x13)
x13 = LeakyReLU(alpha = 0.2)(x13)
			
'''conv14   x13_up and merge with x0_val 256*256*64 to 512*512*64 to 512*512*(64+3) to 512*512*3'''
x13_up = UpSampling2D(size = [2, 2], interpolation="nearest")(x13)
x14 = concat_layer()([x13_up, x0_val])
prediction = Conv2D(filters = 3,
			kernel_size = [3, 3],
			strides = [1, 1],
			padding = "same",
			activation = LeakyReLU(alpha = 0.2))(x14)

'''use vgg19'''
base_model = VGG19(include_top = False, weights='imagenet')

'''To freeze the layers'''

for layer in base_model.layers[:]:
	layer.trainable = False

_gt = Lambda(lambda x: preprocess_input(x))(gt)

#_gt = preprocess_input(gt)

_prediction = Lambda(lambda x: preprocess_input(x))(prediction)
#_prediction = preprocess_input(prediction)

comp = Lambda(lambda x: x[0] * (1 - x[2]) + x[1] * x[2])([prediction, gt, mask0])
_comp = Lambda(lambda x: preprocess_input(x))(comp)

vgg_model0 = Model(inputs=base_model.input, outputs=base_model.get_layer('block1_pool').output)
vgg_model1 = Model(inputs=base_model.input, outputs=base_model.get_layer('block2_pool').output)
vgg_model2 = Model(inputs=base_model.input, outputs=base_model.get_layer('block3_pool').output)

pool0_gt = vgg_model0(_gt)
pool0_pre = vgg_model0(_prediction)
pool0_comp = vgg_model0(_comp)

pool1_gt = vgg_model1(_gt)
pool1_pre = vgg_model1(_prediction)
pool1_comp = vgg_model1(_comp)

pool2_gt = vgg_model2(_gt)
pool2_pre = vgg_model2(_prediction)
pool2_comp = vgg_model2(_comp)
predictions = LossLayer()([prediction, gt, mask0, pool0_pre, pool0_gt, pool0_comp, pool1_pre, pool1_gt, pool1_comp, pool2_pre, pool2_gt, pool2_comp])
model = keras.Model(inputs=inputs, outputs=predictions)
'''读入数据,gt split0, mask0 split1, x0_mask split2'''
'''
from picmaker import load_pic
from picmaker import cl_file
from picmaker import check_pic
model.compile(optimizer=keras.optimizers.Adam(lr = 0.0002), loss=None)
load_pic()
model.load_weights("Inpainting13.pkl")
testimg = check_pic()
cl_file()
pre = model.predict(testimg)
import cv2
cv2.imwrite("tt.png", pre[0])
cv2.imwrite("gt.png", testimg[0,:,:,:3])
cv2.imwrite("msk.png", testimg[0,:,:,6:9])
cv2.imwrite("comp.png", (1 - testimg[0,:,:,3:6]) * pre[0] + testimg[0,:,:,3:6] * testimg[0,:,:,:3])
'''
import cv2
inputimg = np.zeros((1, 256, 256, 9))
tmp = cv2.imread("ppp.jpg")
inputimg[0,:,:,:3] = tmp

img_in = cv2.imread("mask.png")
size = img_in.shape

img_out = np.zeros((256, 256, 1))

for i in range(256):
	for j in range(256):
		if((img_in[i][j][0] == 0)):
			img_out[i][j][0] = 1
		else:
			img_out[i][j][0] = 0

inputimg[0,:,:,3:4] = img_out
inputimg[0,:,:,4:5] = img_out
inputimg[0,:,:,5:6] = img_out

inputimg[0,:,:,6:9] = inputimg[0,:,:,:3] * inputimg[0,:,:,3:6]

model.compile(optimizer=keras.optimizers.Adam(lr = 0.0002), loss=None)
present_path = os.getcwd()
os.chdir("E:\\0NaturezzZ\\SchoolWork\\2019Spring\\AI_Intro\\CV_Smallclass\\Project\\Inpainting")
model.load_weights("lj_Inpainting26.pkl")
os.chdir(present_path)
pre = model.predict(inputimg)
outputimg = np.zeros((256, 256 * 3 + 12, 3))
outputimg[:,:256,:] = inputimg[0,:,:,6:9]
outputimg[:,256 + 6: 256 + 6 + 256,:] = inputimg[0,:,:,:3]
outputimg[:,512+12:,:] = pre[0]

cv2.imwrite("final.png", outputimg)
