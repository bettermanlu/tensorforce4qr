#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
from time import time
import datetime
from qr_cnn import QR_CNN
from tensorflow.contrib import learn
import yaml
import cPickle as pkl
import dataset_hdf5
from tensorflow.contrib.learn.python.learn.preprocessing import categorical_vocabulary
from sklearn.decomposition import PCA
from QueryReformulatorEnv import  QueryReformulatorEnv
import unicodedata
# Parameters
# ==================================================
with open('config.yml','r') as ymlfile:
    cfg=yaml.load(ymlfile)


# Data Preparation
# ==================================================

# Load data
print 'Loading queries Dataset...'
t0 = time()
dh5 = dataset_hdf5.DatasetHDF5(cfg['data']['query_dataset_path'])
qi_train = dh5.get_queries(dset='train')
dt_train = dh5.get_doc_ids(dset='train')
qi_valid = dh5.get_queries(dset='valid')
dt_valid = dh5.get_doc_ids(dset='valid')
qi_test = dh5.get_queries(dset='test')
dt_test = dh5.get_doc_ids(dset='test')
print("Loading queries and docs {}".format(time() - t0))
print '%d train examples' % len(qi_train)
print '%d valid examples' % len(qi_valid)
print '%d test examples' % len(qi_test)
#print 'qi_train',qi_train
#print 'dt_train',dt_train


# Build vocabulary
t0 = time()
word2vec_vocab = pkl.load(open(cfg['data']['pretrained_embedding_path'], "rb"))#374557*500
dim_emb_orig = word2vec_vocab.values()[0].shape[0]
print("Loading word2vec vocabulary in {}".format(time()-t0))
categorical_voc=categorical_vocabulary.CategoricalVocabulary()
for key in word2vec_vocab:
	categorical_voc.add(key)
cfg['data']['vocab_size']=len(word2vec_vocab.keys())+1
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=cfg['data']['max_words_input'],
			   vocabulary=categorical_voc)
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))

W=np.array(word2vec_vocab.values(), dtype='f')
if cfg['data']['embedding_dim'] < dim_emb_orig:
	pca = PCA(n_components=cfg['data']['embedding_dim'], copy=False, whiten=True)
	W = pca.fit_transform(W)
W0=np.random.rand(1,cfg['data']['embedding_dim'])*0.001# for the unknown_token="<UNK>" in CategoricalVocabulary()
W=np.concatenate((W0, W), axis=0)
query_x= np.array(list(vocab_processor.fit_transform(qi_train)))


#load env data

env = QueryReformulatorEnv(cfg['data']['base_path'],dset='train',is_train=True,verbose=True)
[expanded_queries, reward]=env.train_samples(len(qi_train))
expand_text=[]
for i in range(len(expanded_queries)):
	split=cfg['data']['max_words_input']
	expanded=' '.join(expanded_queries[i][split:])
	expandedcode=unicodedata.normalize('NFKD', expanded).encode('ascii', 'ignore')
	expand_text.append(qi_train[i]+' '+expandedcode)
	

terms_vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=cfg['search']['max_terms'],
			   vocabulary=categorical_voc)
terms_x= np.array(list(terms_vocab_processor.fit_transform(expand_text)))
#print terms_x




# Training
# ==================================================
with tf.Graph().as_default():
	session_conf = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
	sess = tf.Session(config=session_conf)
	with sess.as_default():
		sentence_length=cfg['data']['max_words_input']
		expand_length=cfg['search']['max_terms']
		#vocab_size=cfg['data']['vocab_size']
		embedding_dim = cfg['data']['embedding_dim']
		
		filter_sizes_query= cfg['model']['cnn']['filter_sizes_query']
		num_filters_query=cfg['model']['cnn']['num_filters_query']
		num_dense_nodes_query = cfg['model']['cnn']['num_dense_nodes_query']
		num_filters_terms = cfg['model']['cnn']['num_filters_terms']
		filter_sizes_terms =  cfg['model']['cnn']['filter_sizes_terms']
		num_dense_nodes_terms = cfg['model']['cnn']['num_dense_nodes_terms']
		neg_entropy_lambda= cfg['model']['loss']['neg_entropy_lambda']
		alpha=cfg['model']['loss']['alpha']
		cnn=QR_CNN(W,sentence_length,embedding_dim,expand_length,
		           filter_sizes_query,num_filters_query,num_dense_nodes_query,
		           filter_sizes_terms, num_filters_terms, num_dense_nodes_terms,
		           neg_entropy_lambda,alpha)
		# Define Training procedure
		global_step = tf.Variable(0, name="global_step", trainable=False)
		optimizer = tf.train.AdamOptimizer(1e-3)
		grads_and_vars = optimizer.compute_gradients(cnn.loss)
		train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
		
		# Keep track of gradient values and sparsity (optional)
		grad_summaries = []
		for g, v in grads_and_vars:
			if g is not None:
				grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
				sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
				grad_summaries.append(grad_hist_summary)
				grad_summaries.append(sparsity_summary)
		grad_summaries_merged = tf.summary.merge(grad_summaries)
		
		# Output directory for models and summaries
		timestamp = str(int(time()))
		out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
		print("Writing to {}\n".format(out_dir))
		
		# Summaries for loss and accuracy
		loss_summary = tf.summary.scalar("loss", cnn.loss)
		# Train Summaries
		train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
		train_summary_dir = os.path.join(out_dir, "summaries", "train")
		train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
		
		# Dev summaries
		dev_summary_op = tf.summary.merge([loss_summary])
		dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
		dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
		'''
		# Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
		checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
		checkpoint_prefix = os.path.join(checkpoint_dir, "model")
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		saver = tf.train.Saver(tf.global_variables(), max_to_keep=cfg['solver']['num_checkpoints'])'''
		
		# Write vocabulary
		#vocab_processor.save(os.path.join(out_dir, "query_vocab"))
		#terms_vocab_processor.save(os.path.join(out_dir, "terms_vocab"))
		
		# Initialize all variables
		sess.run(tf.global_variables_initializer())
		
		def train_step(xa_batch,xb_batch, y_batch):
			"""
			A single training step
			"""
			feed_dict = {
				cnn.input_xa: xa_batch,
				cnn.input_xb: xb_batch,
				cnn.input_y: y_batch
			}#cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
			_, step, summaries, loss = sess.run(
				[train_op, global_step, train_summary_op,cnn.loss],
				feed_dict)
			time_str = datetime.datetime.now().isoformat()
			print("{}: step {}, loss {:g}".format(time_str, step, loss))
			train_summary_writer.add_summary(summaries, step)
		
			
		
			
		# Generate batches
		xa_batch = query_x
		xb_batch= terms_x
		#y_batch=reward
		y_batch=np.ones((10))
		print xa_batch
		print xb_batch
		print y_batch
		
		train_step(xa_batch,xb_batch, y_batch)
		print cnn.x_b_out
		print cnn.x_a_out
		current_step = tf.train.global_step(sess, global_step)
		print current_step
		
