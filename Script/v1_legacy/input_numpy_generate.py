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

def generate_train_numpy_artifact(batch_size=8192, max_seq_len=100, logger=None):
	"""
	Generate numpy artifact for embedding sequence
	: batch_size (int): number of records in each files
	: max_seq_len (int): max length for sequence input, default 100 
	"""
	global cwd, w2v_registry
	input_numpy_path = os.path.join(cwd, 'input_numpy_artifact')
	if not os.path.isdir(input_numpy_path): os.mkdir(input_numpy_path)
	if not gc.isenabled(): gc.enable()

	div, mod = divmod(900000, batch_size)
	n_batch = div + min(mod, 1)

	# Generate ground truth
	if os.path.isfile(os.path.join(input_numpy_path, 'train_age_0.npy')):
		if logger: logger.info('Target label numpy artifacts exist - skip')
	else:
		for npy_path in ['train_idx_shuffle.npy', 'train_age.npy', 'train_gender.npy']:
			with open(os.path.join(input_path, npy_path), 'rb') as f:
				npy = np.load(f)
			for batch_idx in range(n_batch):
				with open(os.path.join(input_numpy_path, '{}_{}.npy'.format(npy_path.split('.')[0], batch_idx)), 'wb') as f:
					np.save(f, npy[batch_idx*batch_size:(batch_idx+1)*batch_size])
		if logger: logger.info('Target label numpy artifacts ready')

	# Generate embedding sequence
	target_list = ['product', 'advertiser']
	for process_step, target in enumerate(target_list):
		existing_last_idx = True if os.path.isfile(os.path.join(input_numpy_path, 'train_seq_last_idx_0.npy')) else False
		if os.path.isfile(os.path.join(input_numpy_path, 'train_{}_seq_0.npy'.format(target))):
			if logger: logger.info('{} embed sequence numpy artifacts exist - skip'.format(target.capitalize()))
		else:
			w2v_model = Word2Vec.load(w2v_registry[target])
			if logger: logger.info('{} w2v model is loaded'.format(target.capitalize()))
			pkl_path = os.path.join(input_path, 'train_{}_id_seq.pkl'.format(target))
			with open(pkl_path, 'rb') as f:
				pkl = pickle.load(f)
			if logger: logger.info('{} sequence is loaded'.format(target.capitalize()))
			for batch_idx in tqdm.tqdm(range(n_batch), desc=target):
				buf = []
				if not existing_last_idx: 
					last_idx = []
				for seq in pkl[batch_idx*batch_size:(batch_idx+1)*batch_size]:
					tmp = np.stack([w2v_model.wv[item] for item in seq[:max_seq_len]], axis=0)                    # (n_step, embed_size)
					n_step, embed_size = tmp.shape
					if not existing_last_idx: 
						last_idx.append(n_step)
					if n_step < max_seq_len:
						tmp = np.concatenate([tmp, np.zeros((max_seq_len-n_step, embed_size))], axis=0)           # (max_seq_len, embed_size)
					buf.append(tmp)
				buf = np.array(buf)
				save_path = os.path.join(input_numpy_path, 'train_{}_seq_{}.npy'.format(target, batch_idx))
				with open(save_path, 'wb') as f:
					np.save(f, buf)
				if not existing_last_idx:
					last_idx = np.array(last_idx) - 1
					save_path = os.path.join(input_numpy_path, 'train_seq_last_idx_{}.npy'.format(batch_idx))
					with open(save_path, 'wb') as f:
						np.save(f, last_idx)
			if logger: logger.info('{} embed sequence numpy artifacts ready'.format(target.capitalize()))

if __name__=='__main__':
	logger = initiate_logger('input_numpy_generate.log')
	batch_size = 8192 if len(sys.argv)<2 else sys.argv[1]
	max_seq_len = 100 if len(sys.argv)<3 else sys.argv[2]
	generate_train_numpy_artifact(batch_size=batch_size, max_seq_len=max_seq_len, logger=logger)



























