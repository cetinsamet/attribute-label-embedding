# -*- coding: utf-8 -*-
# --------------------------------------------------
#
# train.py
#
# Performs zero-shot training 
#
# Written by cetinsamet -*- cetin.samet@metu.edu.tr
# December, 2019
# --------------------------------------------------
import numpy as np
import argparse
import random
import torch
import math
import os

from torch.utils.data import TensorDataset, DataLoader

from classifier import Compatibility, evaluate
from data_loader import Dataset, index_labels

FN = torch.from_numpy


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--mode', type=str)
parser.add_argument('--data_dir', type=str)
parser.add_argument('--optim_type', type=str)
parser.add_argument('--lr', type=float)
parser.add_argument('--wd', type=float)
parser.add_argument('--lr_decay', type=float)
parser.add_argument('--n_epoch', type=int)
parser.add_argument('--batch_size', type=int)
args = parser.parse_args()

if torch.cuda.is_available():
	device = 'cuda'
else: # CUDA IS NOT AVAILABLE
	device = 'cpu'
	import psutil
	n_cpu = psutil.cpu_count()
	n_cpu_to_use = n_cpu // 4
	torch.set_num_threads(n_cpu_to_use)
	os.environ['MKL_NUM_THREADS'] = str(n_cpu_to_use)
	os.environ['KMP_AFFINITY'] = 'compact'

print("DEVICE: %s" % device)

dset 			= Dataset(args.data_dir, args.mode)

x_seen_train	= FN(dset.x_seen_train).to(device)
y_seen_train 	= FN(dset.y_seen_train).to(device)
y_seen_train_ix = FN(index_labels(dset.y_seen_train, dset.seen_classes)).to(device)

x_seen_test 	= FN(dset.x_seen_test).to(device)
y_seen_test 	= FN(dset.y_seen_test).to(device)

x_unseen_test 	= FN(dset.x_unseen_test).to(device)
y_unseen_test 	= FN(dset.y_unseen_test).to(device)
y_unseen_test_ix = FN(index_labels(dset.y_unseen_test, dset.unseen_classes)).to(device)

attrs 			= FN(dset.attributes).to(device)
seen_attrs 		= FN(dset.seen_attributes).to(device)
unseen_attrs 	= FN(dset.unseen_attributes).to(device)

n_seen_train 	= x_seen_train.shape[0]
n_seen_test 	= x_seen_test.shape[0]
n_unseen_test 	= x_unseen_test.shape[0]

seeds = [123]
#seeds = [123, 16, 26, 149, 1995] # <- Train several times randomly
n_trials = len(seeds)
accs = np.zeros([n_trials, args.n_epoch, 4], 'float32')

for trial, seed in enumerate(seeds):

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	clf = Compatibility(d_in = dset.d_ft, d_out = dset.d_attr).to(device) 	# <- classifier
	ce_loss = torch.nn.CrossEntropyLoss(reduction='sum') 					# <- loss

	if args.optim_type == 'adam':
		optimizer = torch.optim.Adam(params 	= clf.parameters(),
									lr 			= args.lr,
									weight_decay= args.wd)
	elif args.optim_type == 'sgd':
		optimizer = torch.optim.SGD(params 		= clf.parameters(),
									lr 			= args.lr,
									weight_decay= args.wd)
	else:
		raise NotImplementedError

	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay) # <- lr_schedular

	data        = TensorDataset(x_seen_train, y_seen_train_ix)
	data_loader	= DataLoader(data, batch_size=args.batch_size, shuffle=True, drop_last=False)

	for epoch_idx in range(args.n_epoch):

		clf.train() # Train mode: ON

		running_loss = 0.

		for x, y in data_loader: # (x, y) <-> (image feature, image label)

			y_ 			= clf(x, seen_attrs) 	# <- forward pass
			batch_loss 	= ce_loss(y_, y)		# <- calculate loss

			optimizer.zero_grad() 	# <- set gradients to zero
			batch_loss.backward()	# <- calculate gradients
			optimizer.step() 		# <- update weights

			running_loss += batch_loss.item() # <- cumulative loss

		scheduler.step() # <- update learning rate params

		epoch_loss = running_loss / n_seen_train # <- calculate epoch loss

		print("Epoch %4d\tLoss : %s" % (epoch_idx + 1, epoch_loss))
		
		if math.isnan(epoch_loss): exit() # if loss is NAN, terminate!

		if (epoch_idx + 1) % 1 == 0:

			clf.eval() # Evaluation mode: ON

			# ----------------------------------------------------------------------------------------------- #
			# ZERO-SHOT ACCURACY
			acc_zsl = evaluate(model = clf,
							   x 	 = x_unseen_test,
							   y 	 = y_unseen_test_ix,
						  	   attrs = unseen_attrs)
			print("Zero-Shot acc            : %f" % acc_zsl)
			# ------------------------------------------------------- #
			# * ----- * ----- * ----- * ----- * ----- * ----- * ----- *
			# ------------------------------------------------------- #
			# GENERALIZED SEEN ACCURACY
			acc_g_seen = evaluate(model	= clf,
								  x 	= x_seen_test,
								  y 	= y_seen_test,
								  attrs = attrs)
			print("Generalized Seen acc     : %f" % acc_g_seen)
			# ------------------------------------------------------- #
			# * ----- * ----- * ----- * ----- * ----- * ----- * ----- *
			# ------------------------------------------------------- #
			# GENERALIZED UNSEEN ACCURACY
			acc_g_unseen = evaluate(model = clf,
									x 	  = x_unseen_test,
									y 	  = y_unseen_test,
									attrs = attrs)
			print("Generalized Unseen acc   : %f" % acc_g_unseen)
			# ------------------------------------------------------- #
			# * ----- * ----- * ----- * ----- * ----- * ----- * ----- *
			# ------------------------------------------------------- #
			# GENERALIZED ZERO-SHOT ACCURACY
			if acc_g_seen + acc_g_unseen == 0.: # avoid divide by zero error!
				h_score = 0.
			else:
				h_score = (2 * acc_g_seen * acc_g_unseen) / (acc_g_seen + acc_g_unseen)
			print("H-Score                  : %f" % h_score)

			accs[trial, epoch_idx, :] = acc_zsl, acc_g_seen, acc_g_unseen, h_score # <- save accuracy values

zsl_mean   = accs[:, :, 0].mean(axis=0)
zsl_std    = accs[:, :, 0].std(axis=0)
gzsls_mean = accs[:, :, 1].mean(axis=0)
gzsls_std  = accs[:, :, 1].std(axis=0)
gzslu_mean = accs[:, :, 2].mean(axis=0)
gzslu_std  = accs[:, :, 2].std(axis=0)
gzslh_mean = accs[:, :, 3].mean(axis=0)
gzslh_std  = accs[:, :, 3].std(axis=0)

print ('Zsl 	:: average: {mean:} +- {std:}'.format(mean=zsl_mean[-1], std=zsl_std[-1]))
print ('Gzsls 	:: average: {mean:} +- {std:}'.format(mean=gzsls_mean[-1], std=gzsls_std[-1]))
print ('Gzslu 	:: average: {mean:} +- {std:}'.format(mean=gzslu_mean[-1], std=gzslu_std[-1]))
print ('Gzslh 	:: average: {mean:} +- {std:}'.format(mean=gzslh_mean[-1], std=gzslh_std[-1]))
