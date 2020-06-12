import os
import sys
import numpy as np 
import pandas as pd
import logging
import gc
import tqdm
import pickle
import json
import time
import tempfile
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, roc_auc_score

import torch
from torch import nn
import torch.nn.functional as F

from data_loader_v2 import data_loader_v2, wv_loader_v2
from clf_lstm import Multi_Seq_LSTM_Attn_Conv_Classifier

cwd = os.getcwd()
train_path = os.path.join(cwd, 'train_artifact')
test_path = os.path.join(cwd, 'test_artifact')
input_path = os.path.join(cwd, 'input_artifact')
input_split_path = os.path.join(cwd, 'input_split_artifact')
embed_path = os.path.join(cwd, 'embed_artifact')
model_path = os.path.join(cwd, 'model_artifact')

def initiate_logger(log_path):
	"""
	Initialize a logger with file handler and stream handler
	"""
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s %(levelname)-s: %(message)s', datefmt='%H:%M:%S')
	fh = logging.FileHandler(log_path)
	fh.setLevel(logging.INFO)
	fh.setFormatter(formatter)
	logger.addHandler(fh)
	sh = logging.StreamHandler(sys.stdout)
	sh.setLevel(logging.INFO)
	sh.setFormatter(formatter)
	logger.addHandler(sh)
	logger.info('===================================')
	logger.info('Begin executing at {}'.format(time.ctime()))
	logger.info('===================================')
	return logger

def get_torch_module_num_of_parameter(model):
	"""
	Get # of parameters in a torch module.
	"""
	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	return params

def train(model, task, y_list, x_list, checkpoint_dir, checkpoint_prefix, device, batch_size=512, max_seq_len=100, lr=1e-3, resume_surfix=None, logger=None):
	"""
	: model - torch.nn.module: model to be trained
	: task - list[tuple(int,list[int])]: epoch + file to train
	: y_list - list[str]: list of y variables
	: x_list - list[str]: list of x variables to generate embed sequence for
	: checkpoint_dir - str: path to checkpoint directory
	: checkpoint_prefix - str: prefix of checkpoint file
	: device - torch.device: device to train the model
	: batch_size - int: size of mini batch
	: max_seq_len - int: max length for sequence input, default 100 
	: lr - float: learning rate for Adam, default 1e-3
	: resume_surfix - str: model to reload if not training from scratch
	"""
	global input_split_path, embed_path
	if not gc.isenabled(): gc.enable

	# Check checkpoint directory
	if not os.path.isdir(checkpoint_dir): os.mkdir(checkpoint_dir)

	# Initiate word vector host
	wv = wv_loader_v2(x_list, embed_path, max_seq_len=max_seq_len)
	if logger: logger.info('Word vector host ready')

	# Load model if not train from scratch
	if resume_surfix is not None:
		model_artifact_path = os.path.join(checkpoint_dir, '{}_{}.pth'.format(checkpoint_prefix, resume_surfix))
		model.load_state_dict(torch.load(model_artifact_path))
		if logger: logger.info('Model loaded from {}'.format(model_artifact_path))

	# Set up loss function and optimizer
	model.to(device)
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=0, threshold=1e-5, threshold_mode='abs')

	# Main Loop
	for epoch, file_idx_list in task:
		if logger:
			logger.info('=========================')
			logger.info('Processing Epoch {}/{}'.format(epoch, task[-1][0]))
			logger.info('=========================')

		# Train model
		model.train()
		train_age_loss, train_gender_loss, train_n_batch = 0, 0, 0

		for split_idx in file_idx_list:
			dl = data_loader_v2(wv, y_list, x_list, input_split_path, split_idx, batch_size=batch_size, shuffle=True)
			it = iter(dl)
			while True:
				try:
					yl, xl, x_seq_len = next(it)
					y_age = yl[0].to(device)
					y_gender = yl[1].to(device)
					x = [i.to(device) for i in xl] + [x_seq_len]

					optimizer.zero_grad()
					yp = F.softmax(model(*x), dim=1)
					l_age = loss_fn(yp[0],y_age)
					l_gender = loss_fn(yp[1], y_gender)
					loss = l_age + l_gender

					loss.backward()
					torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100)
					optimizer.step()

					train_age_loss += l_age.item()
					train_gender_loss += l_gender.item()
					train_n_batch += 1

				except StopIteration:
					break

				except Exception as e:
					if logger: logger.error(e)
					return 

			del dl, it
			_ = gc.collect()

			if logger:
				logger.info('Epoch {}/{} - File {}/9 Done - Train Loss - Age: {:.6f}, Gender: {:.6f}'.format(epoch, task[-1][0], split_idx, train_age_loss/train_n_batch, train_gender_loss/train_n_batch))

			# Save model state dict
			ck_file_name = '{}_{}_{}.pth'.format(checkpoint_prefix, epoch, split_idx)
			ck_file_path = os.path.join(checkpoint_dir, ck_file_name)
		
			torch.save(model.state_dict(), ck_file_path)

		torch.cuda.empty_cache()

		# Evaluate model
		model.eval()
		test_age_loss, test_gender_loss, test_n_batch = 0, 0, 0
		true_age, pred_age, true_gender, pred_gender = [], [], [], []

		with torch.no_grad():
			dl = data_loader_v2(wv, y_list, x_list, input_split_path, 10, batch_size=batch_size, shuffle=True)
			it = iter(dl)
			while True:
				try:
					yl, xl, x_seq_len = next(it)
					y_age = yl[0].to(device)
					y_gender = yl[1].to(device)
					x = [i.to(device) for i in xl] + [x_seq_len-1]
					yp = F.softmax(model(*x), dim=1)
					l_age = loss_fn(yp[0],y_age)
					l_gender = loss_fn(yp[1], y_gender)
					loss = l_age + l_gender

					pred_age.extend(list(yp[0].cpu().detach().numpy()))
					true_age.extend(list(y_age.cpu().detach().numpy()))
					pred_gender.extend(list(yp[1].cpu().detach().numpy()))
					true_gender.extend(list(y_gender.cpu().detach().numpy()))

					test_age_loss += l_age.item()
					test_gender_loss += l_gender.item()
					test_n_batch += 1

				except StopIteration:
					break

				except Exception as e:
					if logger: logger.error(e)
					return 

			del dl, it
			_ = gc.collect()

		pred_age = np.argmax(np.array(pred_age), 1)
		true_age = np.array(true_age).reshape((-1,))
		acc_age = accuracy_score(true_age, pred_age)

		pred_gender = np.argmax(np.array(pred_gender), 1)
		true_gender = np.array(true_gender).reshape((-1,))
		acc_gender = accuracy_score(true_gender, pred_gender)

		del pred_age, true_age, pred_gender, true_gender
		_ = gc.collect()

		if logger:
			logger.info('Epoch {}/{} Done - Age Loss: {:.6f}, Age Accuracy: {:.6f}'.format(epoch, task[-1][0], test_age_loss/test_n_batch, acc_age))
			logger.info('Epoch {}/{} Done - Gender Loss: {:.6f}, Gender Accuracy: {:.6f}'.format(epoch, task[-1][0], test_gender_loss/test_n_batch, acc_gender))

		scheduler.step(acc_gender+acc_age)
		if logger:
			logger.info('Epoch {}/{} - Updated Learning Rate: {:.8f}'.format(epoch, task[-1][0], optimizer.param_groups[0]['lr']))

if __name__=='__main__':
	assert len(sys.argv) in (5, 7)
	end_epoch = int(sys.argv[1])
	batch_size = int(sys.argv[2])
	max_seq_len = int(sys.argv[3])
	lr = float(sys.argv[4])

	if len(sys.argv)==5:
		resume_surfix = None
		task = [(i, np.arange(1,10)) for i in range(1, end_epoch+1)]
	else:
		resume_epoch = int(sys.argv[5])
		resume_file = int(sys.argv[6])
		if resume_file==1:
			resume_surfix = '{}_{}'.format(resume_epoch-1, 9)
			task = [(i, np.arange(1,10)) for i in range(resume_epoch, end_epoch+1)]
		else:
			resume_surfix = '{}_{}'.format(resume_epoch, resume_file-1)
			task = [(resume_epoch, np.arange(resume_file,10))]+[(i, np.arange(1,10)) for i in range(resume_epoch+1, end_epoch+1)]

	task_name = 'train_v2_multiTar_lstm_attn_conv_multiInp'
	checkpoint_dir = os.path.join(model_path, task_name)
	if not os.path.isdir(checkpoint_dir): os.mkdir(checkpoint_dir)
	checkpoint_prefix = task_name
	logger = initiate_logger(os.path.join(checkpoint_dir, '{}.log'.format(task_name)))
	logger.info('Batch Size: {}, Max Sequence Length: {}, Learning Rate: {}'.format(batch_size, max_seq_len, lr))

	y_list = ['age', 'gender']
	x_list = ['creative', 'ad', 'product', 'advertiser']

	DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	logger.info('Device in Use: {}'.format(DEVICE))
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		t = torch.cuda.get_device_properties(DEVICE).total_memory/1024**3
		c = torch.cuda.memory_cached(DEVICE)/1024**3
		a = torch.cuda.memory_allocated(DEVICE)/1024**3
		logger.info('CUDA Memory: Total {:.2f} GB, Cached {:.2f} GB, Allocated {:.2f} GB'.format(t,c,a))

	model = Multi_Seq_LSTM_Attn_Conv_Classifier([128, 128, 128, 128], [192, 192, 192, 192], [10, 2], seq_len=max_seq_len, device=DEVICE)

	logger.info('Model Parameter #: {}'.format(get_torch_module_num_of_parameter(model)))

	train(model, task, y_list, x_list, checkpoint_dir, checkpoint_prefix, DEVICE, 
		batch_size=batch_size, max_seq_len=max_seq_len, lr=lr, resume_surfix=resume_surfix, logger=logger)
	