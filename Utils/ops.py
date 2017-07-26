from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,name="conv2d"):
	with tf.variable_scope(name):
		w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],initializer=tf.truncated_normal_initializer(stddev=stddev))
		
		conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

		biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
		conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

		return conv


def fully_connected(input,batch_size,output_dim,max_out,name='linear'):
    """Creates a fully connected layer
    Args:
      input:input from previous layer
      batch_size: batch_size
      output_dim: output dimension of next layer
      max_out: None is no max_out else max_out pool
      name: name for the layer
        
    Returns:
      Output from the fully connected layer
    """
    with tf.variable_scope(name):
        flat = tf.reshape(input, [batch_size, -1])
        dim = flat.get_shape()[1].value
        weights = tf.get_variable('weights', shape=[dim, output_dim],initializer=tf.truncated_normal_initializer(stddev=0.04))
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.1))
        result = tf.matmul(flat, weights) + biases
        if max_out is None :
            return result
        else:
            tf.get_variable_scope().reuse_variables()
            for i in range(max_out-1):
                weights = tf.get_variable('weights', shape=[dim, output_dim],initializer=tf.truncated_normal_initializer(stddev=0.04))

                biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.1))
                y = tf.matmul(flat, weights) + biases
                result=tf.maximum(result,y)
    return result

def triplet_loss(anchor, positive, negative, alpha):
    """Calculate the triplet loss according to the FaceNet paper
    
    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.
  
    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
        
        basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
      
    return loss