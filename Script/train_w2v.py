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
embed_path = os.path.join(cwd, 'embed_artifact')

# Training corpus for w2v model
corpus_dic = {
	'creative': os.path.join(embed_path, 'embed_train_creative_id_seq.pkl'),
	'ad': os.path.join(embed_path, 'embed_train_ad_id_seq.pkl'),
	'advertiser': os.path.join(embed_path, 'embed_train_advertiser_id_seq.pkl'),
	'product': os.path.join(embed_path, 'embed_train_product_id_seq.pkl')
}

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

def train(target, embed_size, logger=None):
	"""
	Train a Word2Vec Model and save the model artifact
	"""
	global corpus_dic, embed_path
	assert target in corpus_dic

	start = time.time()
	with open(corpus_dic[target], 'rb') as f:
		corpus = pickle.load(f)
	if logger: logger.info('{} corpus is loaded after {:.2f}s'.format(target.capitalize(), time.time()-start))

	model = Word2Vec(sentences=corpus, size=embed_size, window=175, sg=1, hs=1, min_count=1, workers=16)
	if logger: logger.info('{} w2v training is done after {:.2f}s'.format(target.capitalize(), time.time()-start))

	save_path = os.path.join(embed_path, '{}_sg_embed_s{}_'.format(target, embed_size))
	with tempfile.NamedTemporaryFile(prefix=save_path, delete=False) as tmp:
		tmp_file_path = tmp.name
		model.save(tmp_file_path)
	if logger: logger.info('{} w2v model is saved to {} after {:.2f}s'.format(target.capitalize(), tmp_file_path, time.time()-start))

	return tmp_file_path

if __name__=='__main__':
	assert len(sys.argv)==3
	target, embed_size = sys.argv[1], int(sys.argv[2])

	# Set up w2v model registry
	registry_path = os.path.join(embed_path, 'w2v_registry.json')
	if os.path.isfile(registry_path):
		with open(registry_path, 'r') as f:
			w2v_registry = json.load(f)
	else:
		w2v_registry = {}

	logger = initiate_logger('train_w2v.log')

	# Train w2v model if there hasn't been one registered
	if target not in w2v_registry:
		w2v_path = train(target, embed_size, logger=logger)
		w2v_registry[target] = w2v_path
	else:
		logger.info('{} w2v model found, skip'.format(target.capitalize()))
	
	# Save w2v model registry
	with open(registry_path, 'w') as f:
		json.dump(w2v_registry, f)


