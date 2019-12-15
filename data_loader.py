# -*- coding: utf-8 -*-
# --------------------------------------------------
#
# data_loader.py
#
# Loads validation and test splits of zero-shot setting proposed by GBU paper
# GBU paper: https://arxiv.org/pdf/1707.00600.pdf
# Data with proposed split: https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/
#
# Written by cetinsamet -*- cetin.samet@metu.edu.tr
# December, 2019
# --------------------------------------------------
from scipy.io import loadmat
import numpy as np
import os

join = os.path.join


class Dataset():

	def __init__(self, data_dir, mode):
		super().__init__()

		self.attributes = loadmat(join(data_dir, "attributes.mat"))['attributes'].astype('float32')
		### !!!!! 
		#attr_mat_file_name 	= "" #"feats.mat" (?)
		#attr_mat_key_name	= "" #'PredicateMatrix' (?)
		#self.attributes = loadmat(join(data_dir, attr_mat_file_name))[attr_mat_key_name].astype('float32')
		### !!!!! 

		if mode == 'validation':
			
			self.x_seen_train = loadmat(join(data_dir, mode, "train_features.mat"))['train_features'].astype('float32')
			self.y_seen_train = loadmat(join(data_dir, mode, "train_labels.mat"))['train_labels'].ravel()

			self.x_seen_test = loadmat(join(data_dir, mode, "val_seen_features.mat"))['val_seen_features'].astype('float32')
			self.y_seen_test = loadmat(join(data_dir, mode, "val_seen_labels.mat"))['val_seen_labels'].ravel()

			self.x_unseen_test = loadmat(join(data_dir, mode, "val_unseen_features.mat"))['val_unseen_features'].astype('float32')
			self.y_unseen_test = loadmat(join(data_dir, mode, "val_unseen_labels.mat"))['val_unseen_labels'].ravel()			

		elif mode == 'test':

			self.x_seen_train = loadmat(join(data_dir, mode, "trainval_features.mat"))['trainval_features'].astype('float32')
			self.y_seen_train = loadmat(join(data_dir, mode, "trainval_labels.mat"))['trainval_labels'].ravel()

			self.x_seen_test = loadmat(join(data_dir, mode, "test_seen_features.mat"))['test_seen_features'].astype('float32')
			self.y_seen_test = loadmat(join(data_dir, mode, "test_seen_labels.mat"))['test_seen_labels'].ravel()

			self.x_unseen_test = loadmat(join(data_dir, mode, "test_unseen_features.mat"))['test_unseen_features'].astype('float32')
			self.y_unseen_test = loadmat(join(data_dir, mode, "test_unseen_labels.mat"))['test_unseen_labels'].ravel()			

		self.d_ft = self.x_seen_train.shape[1]
		self.d_attr = self.attributes.shape[1]
		self.seen_classes = np.unique(self.y_seen_train)
		self.unseen_classes = np.unique(self.y_unseen_test)
		self.seen_attributes = self.attributes[np.unique(self.y_seen_train)]
		self.unseen_attributes = self.attributes[np.unique(self.y_unseen_test)]
		self.n_all_classes = self.attributes.shape[0]
		self.n_seen_classes = self.seen_attributes.shape[0]
		self.n_unseen_classes = self.unseen_attributes.shape[0]


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
