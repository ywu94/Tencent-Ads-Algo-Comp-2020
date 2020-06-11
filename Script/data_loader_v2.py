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
from gensim.models import Word2Vec, KeyedVectors
import torch

cwd = os.getcwd()
input_split_path = os.path.join(cwd, 'input_split_artifact')
embed_path = os.path.join(cwd, 'embed_artifact')

class wv_loader_v2(object):
	"""
	Host word vector
	"""
	def __init__(self, target_list, embed_path, max_seq_len=100, keep=True, logger=None):
		self.target_list = target_list
		self.max_seq_len = max_seq_len
		self.keep = keep
		self.logger = logger

		if not gc.isenabled(): gc.enable()

		with open(os.path.join(embed_path, 'wv_registry.json'), 'rb') as f:
			self.wv_registry = json.load(f)
		if keep:
			for target in target_list:
				setattr(self, '{}_wv'.format(target), KeyedVectors.load(self.wv_registry[target]))
				if self.logger: logger.info('{} word vector is loaded and persist in current instance'.format(target.capitalize()))

	def embed(self, target, inp_key):
		if self.keep:
			res = [torch.from_numpy(np.stack([getattr(self, '{}_wv'.format(target))[key] for key in key_list[:self.max_seq_len]], axis=0)).float() for key_list in inp_key]
			return res
		if not self.keep:
			wv = KeyedVectors.load(self.wv_registry[target])
			res = [torch.from_numpy(np.stack([wv[key] for key in key_list[:self.max_seq_len]], axis=0)).float() for key_list in inp_key]
			del wv
			_ = gc.collect()
			return res

class data_loader_v2(object):
	"""
	This data loader is tailored to handle splitted input files.
	"""
	def __init__(self, wv, y_list, x_list, input_split_path, split_idx, batch_size=512, shuffle=True, logger=None):
		"""
		: wv - wv_loader_v2: host of word vector
		: y_list - list[str]: list of y variables
		: x_list - list[str]: list of x variables to generate embed sequence for
		: input_split_path: path to directory that stores splitted input files
		: batch_size - int: batch size for yielding data
		: shuffle - bool: whether to shuffle data before yielding
		"""
		self.wv = wv
		self.y_list = y_list
		self.x_list = x_list
		self.split_idx = split_idx
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.logger = logger

		if not gc.isenabled(): gc.enable()

		for y in y_list:
			setattr(self, 'y_{}_prefix'.format(y), os.path.join(input_split_path,'train_{}'.format(y)))
		for x in x_list:
			setattr(self, 'x_{}_prefix'.format(x), os.path.join(input_split_path,'train_{}_id_seq'.format(x)))

		self.y_data_list = self._load_y()
		self.x_data_list = self._load_x()
		self.x_seq_len = np.array([x.shape[0] for x in self.x_data_list[0]])

		assert self.y_data_list[0].shape[0] == self.x_seq_len.shape[0]

		self.len = self.x_seq_len.shape[0]
		div, mod = divmod(self.len, self.batch_size)
		self.n_batch = div + min(mod, 1)

		self.yield_idx = np.arange(self.len)
		if self.shuffle: np.random.shuffle(self.yield_idx)

	def _load_y(self):
		buf = []
		for y in self.y_list:
			with open('{}_{}.npy'.format(getattr(self, 'y_{}_prefix'.format(y)), self.split_idx), 'rb') as f:
				buf.append(torch.from_numpy(np.load(f)).long())
			_ = gc.collect()
		return buf

	def _load_x(self):
		buf = []
		for x in self.x_list:
			with open('{}_{}.pkl'.format(getattr(self, 'x_{}_prefix'.format(x)), self.split_idx), 'rb') as f:
				buf.append(self.wv.embed(x, pickle.load(f)))
			_ = gc.collect()
		return buf

	def __iter__(self):
		self.cur_batch = 0
		return self

	def __next__(self):
		if self.cur_batch >= self.n_batch:
			raise StopIteration
		else:
			if self.logger: self.logger.info('Yielding batch {}/{}'.format(self.cur_batch+1, self.n_batch))
			cur_index = self.yield_idx[self.cur_batch*self.batch_size:(self.cur_batch+1)*self.batch_size]
			res_y = [i[cur_index] for i in self.y_data_list]
			res_x = [torch.nn.utils.rnn.pad_sequence([i[idx] for idx in cur_index], batch_first=True, padding_value=0) for i in self.x_data_list]
			res_x_seq_len = self.x_seq_len[cur_index]
			self.cur_batch += 1
			return res_y, res_x, res_x_seq_len


