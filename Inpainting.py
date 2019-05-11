import os
import numpy
import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer

'''调用形式：MyLayer(初始化列表)(inputs = [img, mask])'''
class MyLayer(Layer):
	
	def __init__(self, output_dim, **kwargs):
		self.supports_masking = True
		self.output_dim = output_dim
		super(MyLayer, self).__init__(**kwargs)
		
	def compute_mask(self, input, input_mask=None):
		return None
		
	def build(self, input_shape):
		self.kernel = self.add_weight(name="kernel",
										shape=(input_shape[1], self.output_dim),
										initializer='uniform',
										trainable=True)
		super(MyLayer, self).build(input_shape)
	
	def call(self, inputs, mask = None):
		normalization = K.mean(inputs[1], axis[1, 2], keepdims = True)
		normalization = K.repeat_elements(normalization, inputs[1].shape[1], axis=1)
		normalization = K.repeat_elements(normalization, inputs[1].shape[2], axis=2)
		
		img_output = K.conv2d(
			(inputs[0] * inputs[1]) / normalization, self.kernel,
			strides=self.strides,
			padding=self.padding,
			data_format=self.data_format,
			dilation_rate=self.dilation_rate
		)
		
		img_output = K.conv2d(
			inputs[1], K.ones(self.kernal.shape),
			strides=self.strides,
			padding=self.padding,
			data_format=self.data_format,
			dilation_rate=self.dilation_rate
		)
		
		mask_output = K.cast(K.greater(mask_output, 0), "float32")
		
		if self.use_bias:
			img_output = K.bias_add(
				img_output,
				self.bias,
				data_format=self.data_format)
				
		if self.activation is not None:
			img_output = self.activation(img_output)
			
		return [img_output, mask_output]
		
	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.output_dim)

'''调用形式: LossLayers(初始化列表)(12个layers的list)'''
class LossLayer(Layer):
	def __init__(self, **kwargs):
		self.is_placeholder = True
		super(CustomVariationalLayer, self).__init__(**kwargs)
		
	def loss_hole(self, mask, gt, predic):
		return self.l1((1 - mask) * gt, (1 - mask) * predic)
		
	def loss_valid(self, mask, gt, predic):
		return self.l1(mask * gt, mask * predic)
		
	#predic, gt, comp 在vgg16跑下来pool1、pool2、pool3的
	def loss_perceptual(self, vgg_predic, vgg_gt, vgg_comp):
		loss = 0.0
		for pre, gt, comp in zip(vgg_predic, vgg_gt, vgg_comp):
			loss += self.l1(pre, gt) + self.l1(comp, gt)
		return loss
	
	#loss_styleout(predict) + loss_stylecomp
	def loss_style(self, vgg_predic, vgg_gt, vgg_comp):
		loss = 0.0
		for pre, gt, comp in zip(vgg_predic, vgg_gt, vgg_comp):
			loss += self.l1(self.gram_matrix(pre), self.gram_matrix(gt))
			loss += self.l1(self.gram_matrix(comp), self.gram_matrix(gt))
		return loss
	
	#平滑度计算，先对遮挡区域适当扩大（因为需要关注和非遮挡区域的平滑度）
	def loss_variation(self, mask, predic):
		kernel = K.ones((3, 3, mask.shape[2], maskshape[2]))
		dilated_holes = K.conv2d(1 - mask, kernal, data_format = "channels_last", padding = "same")
		dilated_holes = K.cast(K.greater(delated_holes, 0), "float32")
		ret = dilated_holes * predic
		return self.l1(ret[:,1:,:,:], ret[:,:-1,:,:]) + self.l1(ret[:,:,1:,:], ret[:,:,:-1,:])
		
	#inputs为list, 应该传入predic, gt, mask, (pool1, pool2, pool3)*(predic, gt, comp) 共12个参数 传了pool1,再传pool2
	def call(self, inputs):
		loss = 0.0;
		loss += loss_hole(inputs[2], inputs[1], inputs[0])
		loss += loss_valid(inputs[2], inputs[1], inputs[0])
		for i in range(3, 10, 3):
			loss += loss_perceptual(inputs[i], inputs[i + 1], inputs[i + 2])
			loss += loss_style(inputs[i], inputs[i + 1], inputs[i + 2])
		loss += loss_variation(inputs[2], inputs[0])
		self.add_loss(loss, inputs = inputs)
		return inputs[0]
		
