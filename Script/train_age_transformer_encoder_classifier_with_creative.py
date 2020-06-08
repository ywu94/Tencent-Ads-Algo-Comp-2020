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

from data_loader import train_data_loader, test_data_loader
from transformer_encoder_classifier import Transformer_Encoder_Classifier

cwd = os.getcwd()
train_path = os.path.join(cwd, 'train_artifact')
test_path = os.path.join(cwd, 'test_artifact')
input_path = os.path.join(cwd, 'input_artifact')
input_split_path = os.path.join(cwd, 'input_split_artifact')
embed_path = os.path.join(cwd, 'embed_artifact')
model_path = os.path.join(cwd, 'model_artifact')

registry_path = os.path.join(embed_path, 'w2v_registry.json')
with open(registry_path, 'r') as f:
	w2v_registry = json.load(f)

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

def train(model, train_inp_tuple, validation_inp_tuple, checkpoint_dir, checkpoint_prefix, device, epoches=5, batch_size=1024, logger=None, epoch_start=0, max_seq_len=100):
	"""
	: model (torch.nn.module): model to be trained
	: train_inp_tuple (list[tuple(str, list[str], list[str])]): list of input for train_data_loader
		: str: path to label data
		: list[str]: list of embedding variables
		: list[str]: list of paths to a pkl file 
	: validation_inp_tuple (list[tuple(str, list[str], list[str])]): list of input for train_data_loader
		: str: path to label data
		: list[str]: list of embedding variables
		: list[str]: list of paths to a pkl file
	: checkpoint_dir (str): path to checkpoint directory
	: checkpoint_prefix (str): prefix of checkpoint file
	: device (str): device to train the model
	: epoches (int): number of epoches to train
	: batch_size (int): size of mini batch
	: epoch_start (int): if = 0 then train a new model, else load an existing model and continue to train, default 0
	: max_seq_len (int): max length for sequence input, default 100 
	"""
	global w2v_registry, model_path
	gc.enable()

	# Check checkpoint directory
	if not os.path.isdir(checkpoint_dir): os.mkdir(checkpoint_dir)

	# Load model if not train from scratch
	if epoch_start != 0:
		model_artifact_path = os.path.join(checkpoint_dir, '{}_{}.pth'.format(checkpoint_prefix, epoch_start))
		model.load_state_dict(torch.load(model_artifact_path))
		if logger: logger.info('Start retraining from epoch {}'.format(epoch_start))

	# Set up loss function and optimizer
	model.to(device)
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters())

	div, mod = divmod(810000, batch_size)
	n_batch_estimate = div + min(mod, 1)

	# Main Loop
	for epoch in range(1+epoch_start, epoches+1+epoch_start):
		if logger:
			logger.info('=========================')
			logger.info('Processing Epoch {}/{}'.format(epoch, epoches+epoch_start))
			logger.info('=========================')

		# Train model
		model.train()
		train_running_loss, train_n_batch = 0, 0

		for index, (label_artifact_path, seq_inp_target, seq_inp_path) in enumerate(train_inp_tuple, start=1):
			train_loader = train_data_loader(label_artifact_path, seq_inp_target, seq_inp_path, w2v_registry, batch_size=batch_size, max_seq_len=max_seq_len)
			train_iterator = iter(train_loader)
			while True:
				try:
					y, x_seq, x_last_idx = next(train_iterator)
					y = torch.from_numpy(y).long().to(device)
					x = []
					for s in x_seq:
						x.append(s.to(device))
					x.append(x_last_idx)
					optimizer.zero_grad()
					yp = F.softmax(model(*x), dim=1)
					loss = loss_fn(yp, y)

					loss.backward()
					torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100)
					optimizer.step()

					train_running_loss += loss.item()
					train_n_batch += 1

					if train_n_batch%100==0 and logger:
						logger.info('Epoch {}/{} - Batch {}/{} Done - Train Loss: {:.6f}'.format(epoch, epoches+epoch_start, train_n_batch, n_batch_estimate, train_running_loss/train_n_batch))
					del x, y, yp, x_seq, x_last_idx
					_ = gc.collect()
					torch.cuda.empty_cache()

				except StopIteration:
					break

			del train_loader, train_iterator
			_ = gc.collect()
			torch.cuda.empty_cache()

			if logger:
				logger.info('Epoch {}/{} - Batch {}/{} Done - Train Loss: {:.6f}'.format(epoch, epoches+epoch_start, train_n_batch, n_batch_estimate, train_running_loss/train_n_batch))

		# Evaluate model
		model.eval()
		test_running_loss, test_n_batch = 0, 0
		true_y, pred_y = [], []

		for index, (label_artifact_path, seq_inp_target, seq_inp_path) in enumerate(validation_inp_tuple, start=1):
			train_loader = train_data_loader(label_artifact_path, seq_inp_target, seq_inp_path, w2v_registry, batch_size=batch_size, max_seq_len=max_seq_len)
			train_iterator = iter(train_loader)
			while True:
				try:
					y, x_seq, x_last_idx = next(train_iterator)
					y = torch.from_numpy(y).long().to(device)
					x = []
					for s in x_seq:
						x.append(s.to(device))
					x.append(x_last_idx)
					yp = F.softmax(model(*x), dim=1)
					loss = loss_fn(yp, y)

					pred_y.extend(list(yp.cpu().detach().numpy()))
					true_y.extend(list(y.cpu().detach().numpy()))

					test_running_loss += loss.item()
					test_n_batch += 1

					del x, y, yp, x_seq, x_last_idx
					_ = gc.collect()
					torch.cuda.empty_cache()

				except StopIteration:
					break

			del train_loader, train_iterator
			_ = gc.collect()
			torch.cuda.empty_cache()

		pred = np.argmax(np.array(pred_y), 1)
		true = np.array(true_y).reshape((-1,))
		acc_score = accuracy_score(true, pred)

		del pred, true, pred_y, true_y
		_ = gc.collect()
		torch.cuda.empty_cache()

		if logger:
			logger.info('Epoch {}/{} Done - Test Loss: {:.6f}, Test Accuracy: {:.6f}'.format(epoch, epoches+epoch_start, test_running_loss/test_n_batch, acc_score))

		# Save model state dict
		ck_file_name = '{}_{}.pth'.format(checkpoint_prefix, epoch)
		ck_file_path = os.path.join(checkpoint_dir, ck_file_name)
		
		torch.save(model.state_dict(), ck_file_path)

if __name__=='__main__':
	assert len(sys.argv)>=5
	epoch_start = int(sys.argv[1])
	epoches = int(sys.argv[2])
	batch_size = int(sys.argv[3])
	max_seq_len = int(sys.argv[4])
	if len(sys.argv)>5:
		train_inp_tuple = [(os.path.join(input_split_path, 'train_age_{}.npy'.format(i)), ['creative'], 
			[os.path.join(input_split_path, 'train_creative_id_seq_{}.pkl'.format(i))]) for i in range(1,10)]
		validation_inp_tuple = [(os.path.join(input_split_path, 'train_age_{}.npy'.format(i)), ['creative'], 
			[os.path.join(input_split_path, 'train_creative_id_seq_{}.pkl'.format(i))]) for i in range(10,11)]
		checkpoint_dir = os.path.join(model_path, 'Transformer_Encoder_Classifier_Creative_Age')
		checkpoint_prefix = 'Transformer_Encoder_Classifier_Creative_Age'
	else:
		train_inp_tuple = [(os.path.join(input_path, 'train_age_tra.npy'), ['product', 'advertiser', 'creative', 'ad'], 
			[os.path.join(input_path, 'train_creative_id_seq_tra.pkl')])]
		validation_inp_tuple = [(os.path.join(input_path, 'train_age_val.npy'), ['product', 'advertiser', 'creative', 'ad'], 
			[os.path.join(input_path, 'train_creative_id_seq_val.pkl')])]
		checkpoint_dir = os.path.join(model_path, 'Transformer_Encoder_Classifier_Creative_Age')
		checkpoint_prefix = 'Transformer_Encoder_Classifier_Creative_Age'

	logger = initiate_logger('Transformer_Encoder_Classifier_Creative_Age.log')
	logger.info('Epoch Start: {}， Epoch to Train: {}, Batch Size: {}, Max Sequence Length: {}'.format(epoch_start, epoches, batch_size, max_seq_len))

	DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	logger.info('Device in Use: {}'.format(DEVICE))
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		t = torch.cuda.get_device_properties(DEVICE).total_memory/1024**3
		c = torch.cuda.memory_cached(DEVICE)/1024**3
		a = torch.cuda.memory_allocated(DEVICE)/1024**3
		logger.info('CUDA Memory: Total {:.2f} GB, Cached {:.2f} GB, Allocated {:.2f} GB'.format(t,c,a))

	model = Transformer_Encoder_Classifier(256, 10, 4, 8, 1024, DEVICE).to(DEVICE)
	
	train(model, train_inp_tuple, validation_inp_tuple, checkpoint_dir, checkpoint_prefix, DEVICE, 
		epoches=epoches, batch_size=batch_size, logger=logger, epoch_start=epoch_start, max_seq_len=max_seq_len)