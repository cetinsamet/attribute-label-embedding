# -*- coding: utf-8 -*-
# --------------------------------------------------
#
# classifier.py
#
# - Attribute Label Embedding (ALE) compatibility function 
# - Normalized Zero-Shot evaluation
#
# Written by cetinsamet -*- cetin.samet@metu.edu.tr
# December, 2019
# --------------------------------------------------
import torch
import torch.nn.functional as F
import torch.nn as nn


class Compatibility(nn.Module):
	""" Attribute Label Embedding (ALE) compatibility function """
	
	def __init__(self, d_in, d_out):
		super().__init__()
		self.layer  = nn.Linear(d_in, d_out, True)
	
	def forward(self, x, s):
		x = self.layer(x)
		x = F.linear(x, s)
		return x

def evaluate(model, x, y, attrs):
	""" Normalized Zero-Shot Evaluation """

	classes = torch.unique(y)
	n_class = len(classes)
	t_acc   = 0.
	y_ 		= torch.argmax(model(x, attrs), dim=1)
	
	for _class in classes:
		idx_sample	= [i for i, _y in enumerate(y) if _y==_class]
		n_sample 	= len(idx_sample)
		
		y_sample_  	= y_[idx_sample]
		y_sample   	= y[idx_sample].long()

		scr_sample	= torch.sum(y_sample_ == y_sample).item()
		acc_sample	= scr_sample / n_sample
		t_acc   	+= acc_sample
	
	acc = t_acc / n_class
	return acc
