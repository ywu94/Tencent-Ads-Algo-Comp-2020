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
from clf_lstm import Multi_Seq_LSTM_Classifier

cwd = os.getcwd()
train_path = os.path.join(cwd, 'train_artifact')
test_path = os.path.join(cwd, 'test_artifact')
input_path = os.path.join(cwd, 'input_artifact')
input_split_path = os.path.join(cwd, 'input_split_artifact')
embed_path = os.path.join(cwd, 'embed_artifact')
model_path = os.path.join(cwd, 'model_artifact')
output_path = os.path.join(cwd, 'output_artifact')

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

def pred(model, x_list, checkpoint_dir, checkpoint_prefix, output_dir, output_prefix, device, load_surfix, batch_size=512, max_seq_len=100, logger=None):
	"""
	: model - torch.nn.module: model to be trained
	: x_list - list[str]: list of x variables to generate embed sequence for
	: checkpoint_dir - str: path to checkpoint directory
	: checkpoint_prefix - str: prefix of checkpoint file
	: output_dir - str: path to output directory
	: output_prefix - str: prefix of output file
	: device - torch.device: device to train the model
	: load_surfix- - str: model artifact to load
	: batch_size - int: size of mini batch
	: max_seq_len - int: max length for sequence input, default 100
	"""
	global input_split_path, embed_path
	if not gc.isenabled(): gc.enable

	# Initiate word vector host
	wv = wv_loader_v2(x_list, embed_path, max_seq_len=max_seq_len)
	if logger: logger.info('Word vector host ready')

	# Load model 
	model_artifact_path = os.path.join(checkpoint_dir, '{}_{}.pth'.format(checkpoint_prefix, load_surfix))
	model.load_state_dict(torch.load(model_artifact_path))
	if logger: logger.info('Model loaded from {}'.format(model_artifact_path))
	model.to(device)
	model.eval()

	# Main Loop
	pred_y = []

	for file_idx in np.arange(1,11):
		with torch.no_grad():
			dl = data_loader_v2(wv, [], x_list, input_split_path, file_idx, batch_size=batch_size, shuffle=False, train=False)
			it = iter(dl)
			while True:
				try:
					_, xl, x_seq_len = next(it)
					x = [i.to(device) for i in xl] + [x_seq_len-1]
					yp = F.softmax(model(*x), dim=1)
					pred_y.extend(list(yp.cpu().detach().numpy()))

				except StopIteration:
					break

				except Exception as e:
					if logger: logger.error(e)
					return 

			del dl, it
			_ = gc.collect()

		if logger:
			logger.info('File {}/10 done with {}'.format(file_idx, model_artifact_path))

	pred = np.array(pred_y)

	save_path = os.path.join(output_dir, '{}_{}.npy'.format(output_prefix, load_surfix))
	with open(save_path, 'wb') as f:
		np.save(f, pred)

	if logger: 
		logger.info('Prediction result is saved to {}'.format(save_path))


if __name__=='__main__':
	assert len(sys.argv)==4
	batch_size = int(sys.argv[1])
	max_seq_len = int(sys.argv[2])
	load_surfix = sys.argv[3]

	task_name = 'train_v2_age_lstm_multiInp'

	checkpoint_dir = os.path.join(model_path, task_name)
	if not os.path.isdir(checkpoint_dir): os.mkdir(checkpoint_dir)
	checkpoint_prefix = task_name

	pred_task_name = 'pred_age_lstm_multiInp'

	output_dir = os.path.join(output_path, pred_task_name)
	if not os.path.isdir(output_dir): os.mkdir(output_dir)
	output_prefix = pred_task_name
	
	logger = initiate_logger(os.path.join(output_dir, '{}.log'.format(pred_task_name)))

	x_list = ['creative', 'ad', 'product', 'advertiser']

	DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	logger.info('Device in Use: {}'.format(DEVICE))
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		t = torch.cuda.get_device_properties(DEVICE).total_memory/1024**3
		c = torch.cuda.memory_cached(DEVICE)/1024**3
		a = torch.cuda.memory_allocated(DEVICE)/1024**3
		logger.info('CUDA Memory: Total {:.2f} GB, Cached {:.2f} GB, Allocated {:.2f} GB'.format(t,c,a))

	model = Multi_Seq_LSTM_Classifier([128, 128, 128, 128], [128, 128, 128, 128], 10)

	pred(model, x_list, checkpoint_dir, checkpoint_prefix, output_dir, output_prefix, DEVICE, load_surfix, 
		batch_size=batch_size, max_seq_len=max_seq_len, logger=logger)