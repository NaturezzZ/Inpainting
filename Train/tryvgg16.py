
import inspect
import os
 
import numpy as np
import tensorflow as tf
import time
 
VGG_MEAN = [103.939, 116.779, 123.68]
 
 
class Vgg16:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            print path
	# 加载网络权重参数
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")
 
    def build(self, bgr):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """
        start_time = time.time()
        print("build model started")
 
        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')
 
        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')
 
        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')
 
        print("build model finished: %ds" % (time.time() - start_time))
 
    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
 
    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
 
    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)
 
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
 
            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)
 
            relu = tf.nn.relu(bias)
            return relu
 
    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])
 
            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)
 
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
 
            return fc
 
    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")
 
    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")
 
    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")