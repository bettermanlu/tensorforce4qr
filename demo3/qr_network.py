import tensorflow as tf
import numpy as np


class QR_CNN(object):
	def __init__(self, W,sentence_length,embedding_dim,expand_length,
		           filter_sizes_query,num_filters_query,num_dense_nodes_query,
		           filter_sizes_terms, num_filters_terms, num_dense_nodes_terms,
		           neg_entropy_lambda,alpha):
		# Placeholders for input, output and dropout
		self.input_xa = tf.placeholder(tf.int32, [None, sentence_length], name="input_xa")
		self.input_xb = tf.placeholder(tf.int32, [None, expand_length], name="input_xb")
		self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
		#self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
		# build the embedding layer
		
		query_embedding,terms_embedding=self.embedding_block(W)
		# Create a convolution + maxpool layer for each filter size
		print 'build the query CNN'
		x_a = self.conv_block(query_embedding,filter_sizes_query, num_filters_query, 'a')
		print 'build the candidate terms CNN'
		x_b = self.conv_block(terms_embedding, filter_sizes_terms, num_filters_terms, 'b')
		
		x_a_vector = tf.concat([x_a,x_b], 1)
		print 'combined conv results for query network:', np.shape(x_a_vector),
		x_a_vector = tf.layers.flatten(x_a_vector)
		print np.shape(x_a_vector)
		self.x_a_out = self.fc_block(x_a_vector, num_dense_nodes_query, 'a')
		
		mean_xb=tf.reduce_mean(x_b,axis=1,keep_dims=True)
		print 'mean of conv results for candidate terms network:', np.shape(mean_xb)
		x_b_vector = tf.concat([x_a, mean_xb], 1)
		print 'combined conv results for candidate terms network:', np.shape(x_b_vector),
		x_b_vector = tf.layers.flatten(x_b_vector)
		print np.shape(x_b_vector)
		self.x_b_out = self.fc_block(x_b_vector, num_dense_nodes_terms, 'b')
		self.loss=self.loss_block(self.x_a_out,self.x_b_out,neg_entropy_lambda, alpha)
	
	def embedding_block(self,W):
		with tf.name_scope("embedding"):
			query_embedding = tf.nn.embedding_lookup(np.array(W,dtype='f'), self.input_xa)
			query_embedding_expanded = tf.expand_dims(query_embedding, -1)  # (None, 15, 500, 1)
			print query_embedding_expanded
			terms_embedding = tf.nn.embedding_lookup(np.array(W,dtype='f'), self.input_xb)
			terms_embedding_expanded = tf.expand_dims(terms_embedding, -1)  # (None, 30, 500, 1)
			print terms_embedding_expanded
		return query_embedding_expanded,terms_embedding_expanded
		
	
	def conv_block(self,input_tensor, filter_sizes, num_filters, branch, strides=2):
		print 'conv block'
		print 'input data:',np.shape(input_tensor)
		with tf.name_scope("conv-maxpool-%s" % branch):
			conv_name_base = 'conv_' + branch
			pool_name_base = 'max_pooling_' + branch
			
			x = tf.layers.conv2d(input_tensor, num_filters[0], filter_sizes[0], strides,padding='same' ,name=conv_name_base+str(1) )
			print 'output of conv1:', np.shape(x)
			x = tf.nn.relu(x)
			x = tf.layers.max_pooling2d(x, (2, 2), 2, name=pool_name_base+str(1))
			print 'output of maxpool1:',np.shape(x)
			x = tf.layers.conv2d(x, num_filters[1], filter_sizes[1], strides, padding='same',
			                     name=conv_name_base + str(2))
			print 'output of conv2:',np.shape(x)
			x = tf.nn.relu(x)
			x = tf.layers.max_pooling2d(x, (2, 2), 2, name=pool_name_base + str(2))
			print 'output of conv2:', np.shape(x)
		return x
		
	
	
	def fc_block(self,input_tensor, num_dense_nodes, branch):
		print 'fc block'
		with tf.name_scope("fc-%s"   % branch):
			units1, units2 = num_dense_nodes  # 256,256
			fc_name_base = 'fc_' + branch
			x = tf.layers.dense(input_tensor, units1,name=fc_name_base+'1')
			print 'output of fc1:',np.shape(x)
			x = tf.nn.tanh(x)
			x = tf.layers.dense(x, units2,name=fc_name_base+'2')
			print 'output of fc2:',np.shape(x)
			x = tf.nn.sigmoid(x)
		return x
		
	
	def loss_block(self,x_a_out,x_b_out,neg_entropy_lambda,alpha):
		reward_true=self.input_y
		reward_pred=x_b_out
		prop_pred=x_a_out
		print 'reward_true',np.shape(reward_true)
		print 'reward_pred', np.shape(reward_pred)
		print 'prop_pred', np.shape(prop_pred)
		with tf.name_scope("loss"):
			loss_a = tf.reduce_sum(tf.subtract(reward_true,reward_pred) * tf.reduce_sum(tf.negative(tf.log(prop_pred))))
			loss_b = alpha * tf.reduce_sum(tf.square(tf.subtract(reward_true, reward_pred)))
			regulation = -neg_entropy_lambda * tf.reduce_sum(tf.matmul(tf.transpose(prop_pred), tf.log(prop_pred)))
			loss = loss_a + loss_b + regulation
		print "loss",np.shape(loss)
		return loss
