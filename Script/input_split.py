import os
import sys
import numpy as np 
import pandas as pd
import logging
import gc
import tqdm
import pickle
import time

gc.enable()
cwd = os.getcwd()
train_path = os.path.join(cwd, 'train_artifact')
test_path = os.path.join(cwd, 'test_artifact')
input_path = os.path.join(cwd, 'input_artifact')
embed_path = os.path.join(cwd, 'embed_artifact')

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

def rough_split(logger=None):
	"""
	Split training data (900,000) records into training file and validation file using ratio 9:1.
	"""
	for npy_path in ['train_idx_shuffle.npy', 'train_age.npy', 'train_gender.npy']:
		with open(os.path.join(input_path, npy_path), 'rb') as f:
			npy = np.load(f)
		with open(os.path.join(input_path, '{}_tra.npy'.format(npy_path.split('.')[0])), 'wb') as f:
			np.save(f, npy[:810000])
		with open(os.path.join(input_path, '{}_val.npy'.format(npy_path.split('.')[0])), 'wb') as f:
			np.save(f, npy[810000:])
		if logger: logger.info('{} splitted'.format(npy_path))
	for pkl_path in ['train_creative_id_seq.pkl', 'train_ad_id_seq.pkl', 'train_advertiser_id_seq.pkl', 'train_product_id_seq.pkl']:
		with open(os.path.join(input_path, pkl_path), 'rb') as f:
			pkl = pickle.load(f)
		with open(os.path.join(input_path, '{}_tra.pkl'.format(pkl_path.split('.')[0])), 'wb') as f:
			pickle.dump(pkl[:810000], f)
		with open(os.path.join(input_path, '{}_val.pkl'.format(pkl_path.split('.')[0])), 'wb') as f:
			pickle.dump(pkl[810000:], f)
		if logger: logger.info('{} splitted'.format(pkl_path))

def fine_split(logger=None):
	"""
	Split train data (900,000 records) into 10 files
	Split test data (1,000,000 records) into 10 files
	"""
	input_split_path = os.path.join(cwd, 'input_split_artifact')
	if not os.path.isdir(input_split_path): os.mkdir(input_split_path)
	
	for npy_path in ['train_idx_shuffle.npy', 'train_age.npy', 'train_gender.npy']:
		with open(os.path.join(input_path, npy_path), 'rb') as f:
			npy = np.load(f)
		for i in range(10):
			with open(os.path.join(input_split_path, '{}_{}.npy'.format(npy_path.split('.')[0], i+1)), 'wb') as f:
				np.save(f, npy[i*90000:(i+1)*90000])
		if logger: logger.info('{} splitted'.format(npy_path))
	for pkl_path in ['train_creative_id_seq.pkl', 'train_ad_id_seq.pkl', 'train_advertiser_id_seq.pkl', 'train_product_id_seq.pkl',]:
		with open(os.path.join(input_path, pkl_path), 'rb') as f:
			pkl = pickle.load(f)
		for i in range(10):
			with open(os.path.join(input_split_path, '{}_{}.pkl'.format(pkl_path.split('.')[0], i+1)), 'wb') as f:
				pickle.dump(pkl[i*90000:(i+1)*90000], f)
		if logger: logger.info('{} splitted'.format(pkl_path))

	for npy_path in ['test_idx_shuffle.npy']:
		with open(os.path.join(input_path, npy_path), 'rb') as f:
			npy = np.load(f)
		for i in range(10):
			with open(os.path.join(input_split_path, '{}_{}.npy'.format(npy_path.split('.')[0], i+1)), 'wb') as f:
				np.save(f, npy[i*100000:(i+1)*100000])
		if logger: logger.info('{} splitted'.format(npy_path))
	for pkl_path in ['test_creative_id_seq.pkl', 'test_ad_id_seq.pkl', 'test_advertiser_id_seq.pkl', 'test_product_id_seq.pkl',]:
		with open(os.path.join(input_path, pkl_path), 'rb') as f:
			pkl = pickle.load(f)
		for i in range(10):
			with open(os.path.join(input_split_path, '{}_{}.pkl'.format(pkl_path.split('.')[0], i+1)), 'wb') as f:
				pickle.dump(pkl[i*100000:(i+1)*100000], f)
		if logger: logger.info('{} splitted'.format(pkl_path))


if __name__=='__main__':
	logger = initiate_logger('input_split.log')
	if len(sys.argv)==1:
		rough_split(logger=logger)
	else:
		fine_split(logger=logger)