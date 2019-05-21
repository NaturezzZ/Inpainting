import os
import numpy
import tensorflow as tf
import keras
from keras import backend as K
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
		return tuple(input_shape[0], input_shape[1], input_shape[2], 3)
		
class split1(Layer):
	def call(self, inputs):
		return inputs[:,:,:,3:6]
	def compute_output_shape(self, input_shape):
		return tuple(input_shape[0], input_shape[1], input_shape[2], 3)
	
class split2(Layer):
	def call(self, inputs):
		return inputs[:,:,:,6:9]
	def compute_output_shape(self, input_shape):
		return tuple(input_shape[0], input_shape[1], input_shape[2], 3)
		
class Mask(Layer):
	def call(self, inputs):
		output = inputs[0] * inputs[1]
		return output
	def compute_output_shape(self, input_shape):
		return input_shape[0]
		
class change_value(Layer):
	def call(self, inputs):
		normalization = K.mean(inputs[1], axis = [1, 2], keepdims = True) #均值
		normalization = K.repeat_elements(normalization, inputs[1].shape[1], axis=1)
		normalization = K.repeat_elements(normalization, inputs[1].shape[2], axis=2)
		inputs[0] = inputs[0] / normalization
		return inputs[0]
		
	def compute_output_shape(self, input_shape):
		return input_shape[0]
		
class generate_mask(Conv2D):
	def call(self, inputs):
		mask_output = K.conv2d(
			inputs, K.ones(self.kernal.shape),
			strides=self.strides,
			padding=self.padding,
			data_format=self.data_format,
			dilation_rate=self.dilation_rate
		)
		mask_output = K.cast(K.greater(mask_output, 0), "float32")
		return mask_output
	def compute_output_shape(self, input_shape):
		return tuple(input_shape[0], input_shape[1] // 2, input_shape[2] // 2, self.output_dim)

'''调用形式: LossLayers(初始化列表)(9个layers的list)'''
class LossLayer(Layer):
	def __init__(self, **kwargs):
		self.is_placeholder = True
		super(CustomVariationalLayer, self).__init__(**kwargs)
		
	def loss_hole(self, mask, gt, predic):
		return self.l1((1 - mask) * gt, (1 - mask) * predic)
		
	def loss_valid(self, mask, gt, predic):
		return self.l1(mask * gt, mask * predic)
		
	#predic, gt 在vgg16跑下来pool1、pool2、pool3的
	def loss_perceptual(self, vgg_predic, vgg_gt):
		loss = 0.0
		for pre, gt in zip(vgg_predic, vgg_gt):
			loss += self.l1(pre, gt)
		return loss
	
	#loss_styleout(predict)
	def loss_style(self, vgg_predic, vgg_gt):
		loss = 0.0
		for pre, gt in zip(vgg_predic, vgg_gt):
			loss += self.l1(self.gram_matrix(pre), self.gram_matrix(gt))
		return loss
	
	#平滑度计算，先对遮挡区域适当扩大（因为需要关注和非遮挡区域的平滑度）
	def loss_variation(self, mask, predic):
		kernel = K.ones((3, 3, mask.shape[3], mask.shape[3]))
		dilated_holes = K.conv2d(1 - mask, kernal, data_format = "channels_last", padding = "same")
		dilated_holes = K.cast(K.greater(delated_holes, 0), "float32")
		ret = dilated_holes * predic
		a0 = self.l1(ret[:,1:,:,:], ret[:,:-1,:,:])
		a1 = self.l1(ret[:,:,1:,:], ret[:,:,:-1,:])
		return a0 + a1
		
	#inputs为list, 应该传入predic, gt, mask, (pool0, pool1, pool2)*(predic, gt) 共9个参数 传了pool0,再传pool1
	def call(self, inputs):
		loss = 0.0;
		loss += 6.0 * self.loss_hole(inputs[2], inputs[1], inputs[0])
		loss += self.loss_valid(inputs[2], inputs[1], inputs[0])
		for i in range(3, 9, 2):
			loss += 0.1 * self.loss_perceptual(inputs[i], inputs[i + 1])
			loss += 120.0 * self.loss_style(inputs[i], inputs[i + 1])
		loss += 0.1 * self.loss_variation(inputs[2], inputs[0])
		self.add_loss(loss, inputs = inputs)
		return inputs[0]
		
'''伪UNet 比pconv原论文少了两层(encode 和 decode各少一层) ，
使用BN，Adam， loss函数有一点不同，实现起来更方便（希望结果不要太差，T^T）'''

'''拆分'''
inputs = keras.Input(shape = (512, 512, 9))
gt = split0()(inputs)
mask0 = split1()(inputs)
x0_ma = split2()(inputs)
x0_val = change_value()([x0_ma, mask0])


'''conv1 512*512*3 to 256*256*64'''
x1 = Conv2D(filters = 64,
			kernel_size = [7, 7],
			strides = [2, 2],
			padding = "same",
			activation = "relu")(x0_val)
mask1 = generate_mask(filters = 64,
					kernel_size = [7, 7],
					strides = [2, 2],
					padding = "same")(mask0)
x1_ma = Mask()([x1, mask1])
x1_val = change_value()([x1_ma, mask1])

'''conv2 256*256*64 to 128*128*128'''	
x2 = Conv2D(filters = 128,
			kernel_size = [5, 5],
			strides = [2, 2],
			padding = "same",
			activation = None)(x1_val)
x2 = BatchNormalization(axis = 3, name = "Batch2")(x2)
x2 = ReLU()(x2)
mask2 = generate_mask(filters = 128,
					kernel_size = [5, 5],
					strides = [2, 2],
					padding = "same")(mask1)
x2_ma = Mask()([x2, mask2])
x2_val = change_value()([x2_ma, mask2])

'''conv3 128*128*128 to 64*64*256'''
x3 = Conv2D(filters = 256,
			kernel_size = [5, 5],
			strides = [2, 2],
			padding = "same",
			activation = None)(x2_val)
x3 = BatchNormalization(axis = 3, name = "Batch3")(x3)
x3 = ReLU()(x3)
mask3 = generate_mask(filters = 256,
					kernel_size = [5, 5],
					strides = [2, 2],
					padding = "same")(mask2)
x3_ma = Mask()([x3, mask3])
x3_val = change_value()([x3_ma, mask3])

'''conv4 64*64*256 to 32*32*512'''
x4 = Conv2D(filters = 512,
			kernel_size = [3, 3],
			strides = [2, 2],
			padding = "same",
			activation = None)(x3_val)
x4 = BatchNormalization(axis = 3, name = "Batch4")(x4)
x4 = ReLU()(x4)
mask4 = generate_mask(filters = 512,
					kernel_size = [3, 3],
					strides = [2, 2],
					padding = "same")(mask3)
x4_ma = Mask()([x4, mask4])
x4_val = change_value()([x4_ma, mask4])

'''conv5 32*32*512 to 16*16*512'''
x5 = Conv2D(filters = 512,
			kernel_size = [3, 3],
			strides = [2, 2],
			padding = "same",
			activation = None)(x4_val)
x5 = BatchNormalization(axis = 3, name = "Batch5")(x5)
x5 = ReLU()(x5)
mask5 = generate_mask(filters = 512,
					kernel_size = [3, 3],
					strides = [2, 2],
					padding = "same")(mask4)
x5_ma = Mask()([x5, mask5])
x5_val = change_value()([x5_ma, mask5])

'''conv6 16*16*512 to 8*8*512'''
x6 = Conv2D(filters = 512,
			kernel_size = [3, 3],
			strides = [2, 2],
			padding = "same",
			activation = None)(x5_val)
x6 = BatchNormalization(axis = 3, name = "Batch6")(x6)
x6 = ReLU()(x6)
mask6 = generate_mask(filters = 512,
					kernel_size = [3, 3],
					strides = [2, 2],
					padding = "same")(mask5)
x6_ma = Mask()([x6, mask6])
x6_val = change_value()([x6_ma, mask6])

'''conv7 8*8*512 to 4*4*512'''
x7 = BatchNormalization(axis = 3)(x7)
x7 = Conv2D(filters = 512,
			kernel_size = [3, 3],
			strides = [2, 2],
			padding = "same",
			activation = None)(x6_val)
x7 = BatchNormalization(axis = 3, name = "Batch7")(x7)
x7 = ReLU()(x7)
mask7 = generate_mask(filters = 512,
					kernel_size = [3, 3],
					strides = [2, 2],
					padding = "same")(mask6)
x7_ma = Mask()([x7, mask7])
x7_val = change_value()([x7_ma, mask7])

'''
downsample

upsample
'''

'''conv8   x7_up and merge with x6_val 4*4*512 to 8*8*512 to 8*8*(512+512) to 8*8*512'''
x7_up = UpSampling2D(size = [2, 2], interpolation="nearest")(x7_val)
x8 = tf.concat(x7_up, x6_val, axis = 3)
x8 = Conv2D(filters = 512,
			kernel_size = [3, 3],
			strides = [1, 1],
			padding = "same",
			activation = None)(x8)
x8 = BatchNormalization(axis = 3)(x8)
x8 = LeakyReLU(alpha = 0.2)(x8)
			
'''conv9   x8_up and merge with x5_val 8*8*512 to 16*16*512 to 16*16*(512+512) to 16*16*512'''
x8_up = UpSampling2D(size = [2, 2], interpolation="nearest")(x8)
x9 = tf.concat(x8_up, x5_val, axis = 3)
x9 = Conv2D(filters = 512,
			kernel_size = [3, 3],
			strides = [1, 1],
			padding = "same",
			activation = None)(x9)
x9 = BatchNormalization(axis = 3)(x9)
x9 = LeakyReLU(alpha = 0.2)(x9)
			
'''conv10   x9_up and merge with x4_val 16*16*512 to 32*32*512 to 32*32*(512+512) to 32*32*512'''
x9_up = UpSampling2D(size = [2, 2], interpolation="nearest")(x9)
x10 = tf.concat(x9_up, x4_val, axis = 3)
x10 = Conv2D(filters = 512,
			kernel_size = [3, 3],
			strides = [1, 1],
			padding = "same",
			activation = None)(x10)
x10 = BatchNormalization(axis = 3)(x10)
x10 = LeakyReLU(alpha = 0.2)(x10)
			
'''conv11   x10_up and merge with x3_val 32*32*512 to 64*64*512 to 64*64*(512+256) to 64*64*256'''
x10_up = UpSampling2D(size = [2, 2], interpolation="nearest")(x10)
x11 = tf.concat(x10_up, x3_val, axis = 3)
x11 = Conv2D(filters = 256,
			kernel_size = [3, 3],
			strides = [1, 1],
			padding = "same",
			activation = None)(x11)
x11 = BatchNormalization(axis = 3)(x11)
x11 = LeakyReLU(alpha = 0.2)(x11)

'''conv12   x11_up and merge with x2_val 64*64*256 to 128*128*256 to 128*128*(256+128) to 128*128*128'''
x11_up = UpSampling2D(size = [2, 2], interpolation="nearest")(x11)
x12 = tf.concat(x11_up, x2_val, axis = 3)
x12 = Conv2D(filters = 128,
			kernel_size = [3, 3],
			strides = [1, 1],
			padding = "same",
			activation = None)(x12)
x12 = BatchNormalization(axis = 3)(x12)
x12 = LeakyReLU(alpha = 0.2)(x12)
			
'''conv13   x12_up and merge with x1_val 128*128*128 to 256*256*128 to 256*256*(128+64) to 256*256*64'''
x12_up = UpSampling2D(size = [2, 2], interpolation="nearest")(x12)
x13 = tf.concat(x12_up, x1_val, axis = 3)
x13 = Conv2D(filters = 64,
			kernel_size = [3, 3],
			strides = [1, 1],
			padding = "same",
			activation = None)(x13)
x13 = BatchNormalization(axis = 3)(x13)
x13 = LeakyReLU(alpha = 0.2)(x13)
			
'''conv14   x13_up and merge with x0_val 256*256*64 to 512*512*64 to 512*512*(64+3) to 512*512*3'''
x13_up = UpSampling2D(size = [2, 2], interpolation="nearest")(x13)
x14 = tf.concat(x13_up, x0_val, axis = 3)
prediction = Conv2D(filters = 3,
			kernel_size = [3, 3],
			strides = [1, 1],
			padding = "same",
			activation = LeakyReLU(alpha = 0.2))(x14)
'''
待补充部分：
vgg16(gt)
pool0_gt, pool1_gt, pool2_gt = ???
vgg16(prediction)
pool0_pre, pool1_pre, pool2_pre = ???
'''

'''use vgg19'''
base_model = VGG19(weights='imagenet')

'''To freeze the layers'''
for layer in base_model.layers[:]:
	layer.trainable = False
_gt = gt
_gt = np.expand_dims(_gt, axis  = 0)
_gt = preprocess_input(_gt)

_prediction = prediction
_prediction = np.expand_dims(_prediction, axis  = 0)
_prediction = preprocess_input(_prediction)

vgg_model0 = Model(inputs=base_model.input, outputs=base_model.get_layer('block1_pool').output)
vgg_model1 = Model(inputs=base_model.input, outputs=base_model.get_layer('block2_pool').output)
vgg_model2 = Model(inputs=base_model.input, outputs=base_model.get_layer('block3_pool').output)

pool0_gt = vgg_model0.predict(_gt)
pool0_pre = vgg_model0.predict(_prediction)

pool1_gt = vgg_model1.predic(_gt)
pool1_pre = vgg_model1.predict(_prediction)

pool2_gt = vgg.model2.predict(_gt)
pool2_pre = vgg.model2.predict(_prediction)


predictions = LossLayer()([prediction, gt, mask, pool0_pre, pool0_gt, pool1_pre, pool1_gt, pool2_pre, pool2_gt])

model = keras.Model(inputs=inputs, outputs=prediction)

'''读入数据,gt split0, mask0 split1, x0_mask split2'''
from picmaker import makepic
img = makepic()

'''预训练，找到最好的一次，记录网络权重（if判断找最小值待完善）'''
model.compile(optimizer=keras.optimizers.Adam(lr = 0.0002), loss=None)
model.fit(img, epochs=15, batch_size = 15, shuffle=False, validation_split = 0.1)
model.save_weights("Inpainting0.pkl")


'''再次训练'''
model.compile(optimizer=keras.optimizers.Adam(lr = 0.00006), loss=None)
model.load_weights("Inpainting0.pkl")
'''冻结encoder的BN权值'''
nmlist = []
for i in range(2, 8, 1):
	nmlist.append("Batch%d" % i)
for layer in model.layers:
	layerName = str(layer.name)
	if ((layerName is not None) and (layerName in nmlist)):
		layer.trainable = False

model.fit(img, epochs=30, batch_size = 6, shuffle=False, validation_split = 0.1)
model.save_weights("Inpainting1.pkl")


			
