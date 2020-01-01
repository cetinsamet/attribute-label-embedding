# -*- coding: utf-8 -*-
# --------------------------------------------------
#
# data_loader.py
#
# - Loads validation and test splits of zero-shot setting proposed by GBU paper
# - GBU paper: https://arxiv.org/pdf/1707.00600.pdf
# - Data with proposed split: https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/
#
# Written by cetinsamet -*- cetin.samet@metu.edu.tr
# December, 2019
# --------------------------------------------------
from scipy.io import loadmat
import numpy as np
import os

join = os.path.join


class Dataset():

	def __init__(self, dataset, data_dir, mode):

		self.dataset 	= dataset
		self.data_dir 	= data_dir
		self.mode 		= mode

		self.attr = loadmat(join(self.data_dir, "attributes.mat"))['attributes'].astype('float32')
		### !!!!! 
		#attr_mat_file_name 	= "" #"feats.mat" (?)
		#attr_mat_key_name	= "" #'PredicateMatrix' (?)
		#self.attributes = loadmat(join(data_dir, attr_mat_file_name))[attr_mat_key_name].astype('float32')
		### !!!!! 

		path = join(self.data_dir, self.mode)

		if mode == 'validation':

			self.x_s_train 	= loadmat(join(path, "train_features.mat"))['train_features'].astype('float32')
			self.y_s_train 	= loadmat(join(path, "train_labels.mat"))['train_labels'].ravel().astype('int16')

			self.x_s_test 	= loadmat(join(path, "val_seen_features.mat"))['val_seen_features'].astype('float32')
			self.y_s_test 	= loadmat(join(path, "val_seen_labels.mat"))['val_seen_labels'].ravel().astype('int16')

			self.x_u_test 	= loadmat(join(path, "val_unseen_features.mat"))['val_unseen_features'].astype('float32')
			self.y_u_test 	= loadmat(join(path, "val_unseen_labels.mat"))['val_unseen_labels'].ravel().astype('int16')		

		elif mode == 'test':

			self.x_s_train	= loadmat(join(path, "trainval_features.mat"))['trainval_features'].astype('float32')
			self.y_s_train 	= loadmat(join(path, "trainval_labels.mat"))['trainval_labels'].ravel().astype('int16')

			self.x_s_test 	= loadmat(join(path, "test_seen_features.mat"))['test_seen_features'].astype('float32')
			self.y_s_test 	= loadmat(join(path, "test_seen_labels.mat"))['test_seen_labels'].ravel().astype('int16')

			self.x_u_test 	= loadmat(join(path, "test_unseen_features.mat"))['test_unseen_features'].astype('float32')
			self.y_u_test 	= loadmat(join(path, "test_unseen_labels.mat"))['test_unseen_labels'].ravel().astype('int16')			

		self.d_ft 		= self.x_s_train.shape[1]
		self.d_attr 	= self.attr.shape[1]
		
		self.s_class 	= np.unique(self.y_s_train)
		self.u_class 	= np.unique(self.y_u_test)
		
		self.s_attr 	= self.attr[np.unique(self.s_class)]
		self.u_attr 	= self.attr[np.unique(self.u_class)]
		
		self.check_splits() #check if splits are true

	def check_splits(self):
		
		# ----------------------------------------------------------------- #
		### CLASS
		n_class 	= len(self.attr) 		# num of all classes
		n_s_class 	= len(self.s_class)		# num of seen classes
		n_u_class	= len(self.u_class)		# num of unseen classes
		# ----------------------------------------------------------------- #
		### SAMPLE
		n_s_train 	= len(self.x_s_train)	# num of seen train samples
		n_s_test 	= len(self.x_s_test)	# num of seen test samples
		n_u_test 	= len(self.x_u_test)	# num of unseen test samples
		# ----------------------------------------------------------------- #

		# ----------------------------------------------------------------- #
		### SUN -*- Scene UNderstanding
		if self.dataset == 'SUN':
			assert self.d_attr == 102 and n_class == 717
			if self.mode == 'validation':
				assert n_s_class == 580 and n_u_class == 65
				assert (n_s_train + n_s_test + n_u_test) == 10320
			elif self.mode == 'test':
				assert n_s_class == 645 and n_u_class == 72
				assert n_s_train == 10320 and n_s_test == 2580 and n_u_test == 1440
			else:
				raise ValueError("Mode is INVALID! Try [validation/test]")
		# ----------------------------------------------------------------- #
		### CUB -*- Caltech-UCSD Birds 200
		elif self.dataset == 'CUB':
			assert self.d_attr == 312 and n_class == 200
			if self.mode == 'validation':
				assert n_s_class == 100 and n_u_class == 50
				assert (n_s_train + n_s_test + n_u_test) == 7057
			elif self.mode == 'test':
				assert n_s_class == 150 and n_u_class == 50
				assert n_s_train == 7057 and n_s_test == 1764 and n_u_test == 2967
			else:
				raise ValueError("Mode is INVALID! Try [validation/test]")
		# ----------------------------------------------------------------- #
		### AWA1 -*- Animals With Attributes 1
		elif self.dataset == 'AWA1':
			assert self.d_attr == 85 and n_class == 50
			if self.mode == 'validation':
				assert n_s_class == 27 and n_u_class == 13
				assert (n_s_train + n_s_test + n_u_test) == 19832
			elif self.mode == 'test':
				assert n_s_class == 40 and n_u_class == 10
				assert n_s_train == 19832 and n_s_test == 4958 and n_u_test == 5685
			else:
				raise ValueError("Mode is INVALID! Try [validation/test]")
		# ----------------------------------------------------------------- #
		### AWA2 -*- Animals With Attributes 1
		elif self.dataset == 'AWA2':
			assert self.d_attr == 85 and n_class == 50
			if self.mode == 'validation':
				assert n_s_class == 27 and n_u_class == 13
				assert (n_s_train + n_s_test + n_u_test) == 23527
			elif self.mode == 'test':
				assert n_s_class == 40 and n_u_class == 10
				assert n_s_train == 23527 and n_s_test == 5882 and n_u_test == 7913
			else:
				raise ValueError("Mode is INVALID! Try [validation/test]")
		# ----------------------------------------------------------------- #
		## aPY -*- aPascal & aYahoo 
		elif self.dataset == 'APY':
			assert self.d_attr == 64 and n_class == 32
			if self.mode == 'validation':
				assert n_s_class == 15 and n_u_class == 5
				assert (n_s_train + n_s_test + n_u_test) == 5932
			elif self.mode == 'test':
				assert n_s_class == 20 and n_u_class == 12
				assert n_s_train == 5932 and n_s_test == 1483 and n_u_test == 7924
			else:
				raise ValueError("Mode is INVALID! Try [validation/test]")
		# ----------------------------------------------------------------- #
		else:
			raise ValueError("Dataset is INVALID! Try [SUN/CUB/AWA1/AWA2/APY]")
		# ----------------------------------------------------------------- #
		return

def index_labels(labels, classes, check=True):
	"""
	Indexes labels in classes.

	Arg:
		labels:  [batch_size]
		classes: [n_class]
	"""
	indexed_labels = np.searchsorted(classes, labels)
	if check:
		assert np.all(np.equal(classes[indexed_labels], labels))

	return indexed_labels
