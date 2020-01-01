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
	device_type = 'cuda'
	device 		= torch.device(device_type)
else: # CUDA IS NOT AVAILABLE
	device_type = 'cpu'
	device 		= torch.device(device_type)
	import psutil
	n_cpu = psutil.cpu_count()
	n_cpu_to_use = n_cpu // 4
	torch.set_num_threads(n_cpu_to_use)
	os.environ['MKL_NUM_THREADS'] = str(n_cpu_to_use)
	os.environ['KMP_AFFINITY'] = 'compact'

if args.mode == 'test': verbose = True
else: verbose = False

if verbose:
	print("%s dataset running on %s mode with %s device" % (args.dataset.upper(), args.mode.upper(), device_type.upper()))

dset 		= Dataset(args.dataset, args.data_dir, args.mode)

x_s_train	= FN(dset.x_s_train).to(device)
y_s_train 	= FN(dset.y_s_train).to(device)
y_s_train_ix = FN(index_labels(dset.y_s_train, dset.s_class)).to(device)

x_s_test 	= FN(dset.x_s_test).to(device)
y_s_test 	= FN(dset.y_s_test).to(device)

x_u_test 	= FN(dset.x_u_test).to(device)
y_u_test 	= FN(dset.y_u_test).to(device)
y_u_test_ix = FN(index_labels(dset.y_u_test, dset.u_class)).to(device)

attr 		= FN(dset.attr).to(device)
s_attr 		= FN(dset.s_attr).to(device)
u_attr 		= FN(dset.u_attr).to(device)

n_s_train 	= len(x_s_train)

n_class 	= len(attr)
n_s_class 	= len(s_attr)
n_u_class	= len(u_attr)

if verbose:
	print("Seen train 	:", x_s_train.size())
	print("Seen test 	:", x_s_test.size())
	print("Unseen test 	:", x_u_test.size())
	print("Attrs 		:", attr.size())
	print("Seen Attrs 	:", s_attr.size())
	print("Unseen Attrs	:", u_attr.size())

seeds = [123]
#seeds = [123, 16, 26, 149, 1995] # <- Train several times randomly
n_trials = len(seeds)
accs = np.zeros([n_trials, args.n_epoch, 4], 'float32')

for trial, seed in enumerate(seeds):

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	# init classifier
	clf = Compatibility(d_in 	= dset.d_ft, 
						d_out 	= dset.d_attr).to(device)

	# init loss
	ce_loss = torch.nn.CrossEntropyLoss()

	# init optimizer
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

	# init schedular
	#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay) # <- lr_schedular

	data        = TensorDataset(x_s_train, y_s_train_ix)
	data_loader	= DataLoader(data, batch_size=args.batch_size, shuffle=True, drop_last=False)

	for epoch_idx in range(args.n_epoch):

		clf.train() # Classifer train mode: ON

		running_loss = 0.

		for x, y in data_loader: # (x, y) <-> (image feature, image label)

			y_ 			= clf(x, s_attr)		# <- forward pass
			batch_loss 	= ce_loss(y_, y)		# <- calculate loss

			optimizer.zero_grad() 	# <- set gradients to zero
			batch_loss.backward()	# <- calculate gradients
			optimizer.step() 		# <- update weights

			running_loss += batch_loss.item() * args.batch_size # <- cumulative loss

		#scheduler.step() # <- update schedular

		epoch_loss = running_loss / n_s_train # <- calculate epoch loss

		print("Epoch %4d\tLoss : %s" % (epoch_idx + 1, epoch_loss))
		
		if math.isnan(epoch_loss): continue # if loss is NAN, skip!

		if (epoch_idx + 1) % 1 == 0:

			clf.eval() # Classifier evaluation mode: ON

			# ----------------------------------------------------------------------------------------------- #
			# ZERO-SHOT ACCURACY
			acc_zsl = evaluate(model = clf,
							   x 	 = x_u_test,
							   y 	 = y_u_test_ix,
						  	   attrs = u_attr)
			# ------------------------------------------------------- #
			# * ----- * ----- * ----- * ----- * ----- * ----- * ----- *
			# ------------------------------------------------------- #
			# GENERALIZED SEEN ACCURACY
			acc_g_seen = evaluate(model	= clf,
								  x 	= x_s_test,
								  y 	= y_s_test,
								  attrs = attr)
			# ------------------------------------------------------- #
			# * ----- * ----- * ----- * ----- * ----- * ----- * ----- *
			# ------------------------------------------------------- #
			# GENERALIZED UNSEEN ACCURACY
			acc_g_unseen = evaluate(model = clf,
									x 	  = x_u_test,
									y 	  = y_u_test,
									attrs = attr)
			# ------------------------------------------------------- #
			# * ----- * ----- * ----- * ----- * ----- * ----- * ----- *
			# ------------------------------------------------------- #
			# GENERALIZED ZERO-SHOT ACCURACY
			if acc_g_seen + acc_g_unseen == 0.: # avoid divide by zero error!
				h_score = 0.
			else:
				h_score = (2 * acc_g_seen * acc_g_unseen) / (acc_g_seen + acc_g_unseen)
			# ----------------------------------------------------------------------------------------------- #
			
			accs[trial, epoch_idx, :] = acc_zsl, acc_g_seen, acc_g_unseen, h_score # <- save accuracy values

			if verbose:
				print("Zero-Shot acc            : %f" % acc_zsl)
				print("Generalized Seen acc     : %f" % acc_g_seen)
				print("Generalized Unseen acc   : %f" % acc_g_unseen)
				print("H-Score                  : %f" % h_score)


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
