import tensorflow as tf 
from Utils.ops import *


class NN1(object):
	"""Class definition for NN1 neural network model"""
	def __init__(self, image_size=[220,220,3],batch_size=90):
		self.image_size = image_size
		self.batch_size = batch_size
		
	def build_model(self):
		#input
		images_placeholder=tf.placeholder(tf.float32,shape=[self.batch_size]+self.image_size,name='images_placeholder')
		#conv1
		conv1=tf.nn.relu(conv2d(images_placeholder,64,7,7,2,2,name='conv1'))
		#pool1
		pool1=tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',data_format='NHWC',name='pool1')
		#rnorm1
		rnorm1=tf.nn.lrn(pool1,name='rnorm1')


		#conv2a
		conv2a=conv2d(rnorm1,64,1,1,1,1,name='conv2a')
		#conv2
		conv2=tf.nn.relu(conv2d(conv2a,192,3,3,1,1,name='conv2'))
		#rnorm2
		rnorm2=tf.nn.lrn(conv2,name='rnorm2')
		#pool2
		pool2=tf.nn.max_pool(rnorm2,ksize=[1,3,3,1],strides=[1,1,1,1],padding='SAME',data_format='NHWC',name='pool2')


		#conv3a
		conv3a=conv2d(pool2,192,1,1,1,1,name='conv3a')
		#conv3
		conv3=tf.nn.relu(conv2d(conv3a,384,3,3,1,1,name='conv3'))
		#pool3
		pool3=tf.nn.max_pool(conv3,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',data_format='NHWC',name='pool3')

		#conv4a
		conv4a=conv2d(pool3,384,1,1,1,1,name='conv4a')
		#conv4
		conv4=tf.nn.relu(conv2d(conv4a,256,3,3,1,1,name='conv4'))

		#conv5a
		conv5a=conv2d(conv4,256,1,1,1,1,name='conv5a')
		#conv5
		conv5=tf.nn.relu(conv2d(conv5a,256,3,3,1,1,name='conv5'))
	

		#conv6a
		conv6a=conv2d(conv5,256,1,1,1,1,name='conv6a')
		#conv6
		conv6=tf.nn.relu(conv2d(conv6a,256,3,3,1,1,name='conv6'))
		#pool4
		pool4=tf.nn.max_pool(conv6,ksize=[1,3,3,1],strides=[1,4,4,1],padding='SAME',data_format='NHWC',name='pool3')

		#concat
		concat=tf.concat([pool4],0,name='concat')

		#Flattening the output for fully connected layer
		fc1=fully_connected(concat,self.batch_size,1*32*128,max_out=2,name='fc1')
		
		fc1 = tf.reshape(fc1, [-1,1,32,128])
		
		fc2=fully_connected(fc1,self.batch_size,1*32*128,max_out=2,name='fc2')

		fc2 = tf.reshape(fc2, [-1,1,32,128])

		fc7128=fully_connected(fc2,self.batch_size,1*1*128,max_out=None,name='fc7128')
		probability=tf.constant(0.5)
		fc7128=tf.nn.dropout(fc7128,keep_prob=probability)

		fc7128 = tf.reshape(fc7128, [-1,1,1,128])

		l2=tf.nn.l2_normalize(fc7128,0,name='l2')
			


		return images_placeholder,l2





