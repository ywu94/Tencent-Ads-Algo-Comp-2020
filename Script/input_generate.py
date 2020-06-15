import os
import sys
import numpy as np 
import pandas as pd
import logging
import gc
import tqdm
import pickle
import time

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

def generate(logger=None):
	"""
	Generate input to the model including
	- Shuffled index for train set
	- Shuffled index for test set
	- Creative ID sequence
	- Ad ID sequence
	- Advertiser sequence
	- Product sequence
	"""
	gc.enable()

	cwd = os.getcwd()
	train_path = os.path.join(cwd, 'train_artifact')
	test_path = os.path.join(cwd, 'test_artifact')
	input_path = os.path.join(cwd, 'input_artifact')
	embed_path = os.path.join(cwd, 'embed_artifact')

	# Prepare Shuffled Index
	np.random.seed(1898)

	if not os.path.isfile(os.path.join(input_path, 'train_idx_shuffle.npy')):
		train_idx = np.arange(1, 900001)
		np.random.shuffle(train_idx)
		save_path = os.path.join(input_path, 'train_idx_shuffle.npy')
		with open(save_path, 'wb') as f:
			np.save(f, train_idx)

		test_idx = np.arange(3000001, 4000001)
		np.random.shuffle(test_idx)
		save_path = os.path.join(input_path, 'test_idx_shuffle.npy')
		with open(save_path, 'wb') as f:
			np.save(f, test_idx)
	else:
		with open(os.path.join(input_path, 'train_idx_shuffle.npy'), 'rb') as f:
			train_idx = np.load(f)
		with open(os.path.join(input_path, 'test_idx_shuffle.npy'), 'rb') as f:
			test_idx = np.load(f)

	if logger: logger.info('Shuffled index ready')

	# Prepare Ground Truth for Training
	if not os.path.isfile(os.path.join(input_path, 'train_gender.npy')):
		truth = pd.read_csv(os.path.join(train_path,'user.csv'))
		truth['gender'] = truth['gender'] - 1
		truth['age'] = truth['age'] - 1

		shuffled_truth = truth.iloc[train_idx - 1, :]
		assert list(train_idx)==shuffled_truth['user_id'].values.tolist()

		truth_gender = shuffled_truth['gender'].values
		save_path = os.path.join(input_path, 'train_gender.npy')
		with open(save_path, 'wb') as f:
			np.save(f, truth_gender)

		truth_age = shuffled_truth['age'].values
		save_path = os.path.join(input_path, 'train_age.npy')
		with open(save_path, 'wb') as f:
			np.save(f, truth_age)

		del truth , shuffled_truth, truth_gender, truth_age
		_ = gc.collect()

	if logger: logger.info('Training ground truth data ready')

	# Load Click Log & Ad Data
	cl = pd.concat([pd.read_csv(os.path.join(train_path,'click_log.csv')), pd.read_csv(os.path.join(test_path,'click_log.csv'))])
	cl.sort_values(['user_id', 'time'], inplace=True)
	ad = pd.concat([pd.read_csv(os.path.join(train_path,'ad.csv')), pd.read_csv(os.path.join(test_path,'ad.csv'))])

	# Prepare Creative Sequence Data
	if not os.path.isfile(os.path.join(input_path, 'train_creative_id_seq.pkl')):
		dic = {}
		dic_dedup = {}

		for user, i in tqdm.tqdm(cl[['user_id', 'creative_id']].values, desc='creative id'):
			i = str(i)
			if user in dic:
				dic[user].append(i)
				if i not in dic_dedup[user]:
					dic_dedup[user].append(i)
			else:
				dic[user] = [i]
				dic_dedup[user] = [i]

		train_seq = []
		test_seq = []
		seq_dedup = []

		for user in train_idx:
			train_seq.append(dic[user])
			seq_dedup.append(dic_dedup[user])

		for user in test_idx:
			test_seq.append(dic[user])
			seq_dedup.append(dic_dedup[user])

		save_path = os.path.join(input_path, 'train_creative_id_seq.pkl')
		with open(save_path, 'wb') as f:
			pickle.dump(train_seq, f)

		save_path = os.path.join(input_path, 'test_creative_id_seq.pkl')
		with open(save_path, 'wb') as f:
			pickle.dump(test_seq, f)

		save_path = os.path.join(embed_path, 'embed_train_creative_id_seq.pkl')
		with open(save_path, 'wb') as f:
			pickle.dump(seq_dedup, f)

		del dic, dic_dedup, train_seq, test_seq, seq_dedup
		_ = gc.collect()

	logger.info('Creative ID sequence data ready')

	# Prepare Advertiser Sequence Data
	if not os.path.isfile(os.path.join(input_path, 'train_advertiser_id_seq.pkl')):
		join = ad[['creative_id', 'advertiser_id']].drop_duplicates()
		merge = pd.merge(cl, join, on='creative_id')
		merge.sort_values(['user_id', 'time'], inplace=True)
		assert merge.shape[0]==cl.shape[0]

		dic = {}
		dic_dedup = {}

		for user, i in tqdm.tqdm(merge[['user_id', 'advertiser_id']].values, desc='advertiser id'):
			i = str(i)
			if user in dic:
				dic[user].append(i)
				if i not in dic_dedup[user]:
					dic_dedup[user].append(i)
			else:
				dic[user] = [i]
				dic_dedup[user] = [i]

		train_seq = []
		test_seq = []
		seq_dedup = []

		for user in train_idx:
			train_seq.append(dic[user])
			seq_dedup.append(dic_dedup[user])

		for user in test_idx:
			test_seq.append(dic[user])
			seq_dedup.append(dic_dedup[user])

		save_path = os.path.join(input_path, 'train_advertiser_id_seq.pkl')
		with open(save_path, 'wb') as f:
			pickle.dump(train_seq, f)

		save_path = os.path.join(input_path, 'test_advertiser_id_seq.pkl')
		with open(save_path, 'wb') as f:
			pickle.dump(test_seq, f)

		save_path = os.path.join(embed_path, 'embed_train_advertiser_id_seq.pkl')
		with open(save_path, 'wb') as f:
			pickle.dump(seq_dedup, f)

		del join, merge, dic, dic_dedup, train_seq, test_seq, seq_dedup
		_ = gc.collect()

	logger.info('Advertiser ID sequence data ready')

	# Prepare Product Sequence Data
	if not os.path.isfile(os.path.join(input_path, 'train_product_id_seq.pkl')):
		join = ad[['creative_id', 'product_id']].drop_duplicates()
		merge = pd.merge(cl, join, on='creative_id')
		merge.sort_values(['user_id', 'time'], inplace=True)
		assert merge.shape[0]==cl.shape[0]

		dic = {}
		dic_dedup = {}

		for user, i in tqdm.tqdm(merge[['user_id', 'product_id']].values, desc='product id'):
			i = str(i)
			if user in dic:
				dic[user].append(i)
				if i not in dic_dedup[user]:
					dic_dedup[user].append(i)
			else:
				dic[user] = [i]
				dic_dedup[user] = [i]

		train_seq = []
		test_seq = []
		seq_dedup = []

		for user in train_idx:
			train_seq.append(dic[user])
			seq_dedup.append(dic_dedup[user])

		for user in test_idx:
			test_seq.append(dic[user])
			seq_dedup.append(dic_dedup[user])

		save_path = os.path.join(input_path, 'train_product_id_seq.pkl')
		with open(save_path, 'wb') as f:
			pickle.dump(train_seq, f)

		save_path = os.path.join(input_path, 'test_product_id_seq.pkl')
		with open(save_path, 'wb') as f:
			pickle.dump(test_seq, f)

		save_path = os.path.join(embed_path, 'embed_train_product_id_seq.pkl')
		with open(save_path, 'wb') as f:
			pickle.dump(seq_dedup, f)

		del join, merge, dic, dic_dedup, train_seq, test_seq, seq_dedup
		_ = gc.collect()

	logger.info('Product ID sequence data ready')

	# Prepare Ad Sequence Data
	if not os.path.isfile(os.path.join(input_path, 'train_ad_id_seq.pkl')):
		join = ad[['creative_id', 'ad_id']].drop_duplicates()
		merge = pd.merge(cl, join, on='creative_id')
		merge.sort_values(['user_id', 'time'], inplace=True)
		assert merge.shape[0]==cl.shape[0]

		dic = {}
		dic_dedup = {}

		for user, i in tqdm.tqdm(merge[['user_id', 'ad_id']].values, desc='ad id'):
			i = str(i)
			if user in dic:
				dic[user].append(i)
				if i not in dic_dedup[user]:
					dic_dedup[user].append(i)
			else:
				dic[user] = [i]
				dic_dedup[user] = [i]

		train_seq = []
		test_seq = []
		seq_dedup = []

		for user in train_idx:
			train_seq.append(dic[user])
			seq_dedup.append(dic_dedup[user])

		for user in test_idx:
			test_seq.append(dic[user])
			seq_dedup.append(dic_dedup[user])

		save_path = os.path.join(input_path, 'train_ad_id_seq.pkl')
		with open(save_path, 'wb') as f:
			pickle.dump(train_seq, f)

		save_path = os.path.join(input_path, 'test_ad_id_seq.pkl')
		with open(save_path, 'wb') as f:
			pickle.dump(test_seq, f)

		save_path = os.path.join(embed_path, 'embed_train_ad_id_seq.pkl')
		with open(save_path, 'wb') as f:
			pickle.dump(seq_dedup, f)

		del join, merge, dic, dic_dedup, train_seq, test_seq, seq_dedup
		_ = gc.collect()

	logger.info('Ad ID sequence data ready')

	# Prepare Industry Sequence Data
	if not os.path.isfile(os.path.join(input_path, 'train_industry_id_seq.pkl')):
		join = ad[['creative_id', 'industry']].drop_duplicates()
		merge = pd.merge(cl, join, on='creative_id')
		merge.sort_values(['user_id', 'time'], inplace=True)
		assert merge.shape[0]==cl.shape[0]

		dic = {}
		dic_dedup = {}

		for user, i in tqdm.tqdm(merge[['user_id', 'industry']].values, desc='industry id'):
			i = str(i)
			if user in dic:
				dic[user].append(i)
				if i not in dic_dedup[user]:
					dic_dedup[user].append(i)
			else:
				dic[user] = [i]
				dic_dedup[user] = [i]

		train_seq = []
		test_seq = []
		seq_dedup = []

		for user in train_idx:
			train_seq.append(dic[user])
			seq_dedup.append(dic_dedup[user])

		for user in test_idx:
			test_seq.append(dic[user])
			seq_dedup.append(dic_dedup[user])

		save_path = os.path.join(input_path, 'train_industry_id_seq.pkl')
		with open(save_path, 'wb') as f:
			pickle.dump(train_seq, f)

		save_path = os.path.join(input_path, 'test_industry_id_seq.pkl')
		with open(save_path, 'wb') as f:
			pickle.dump(test_seq, f)

		save_path = os.path.join(embed_path, 'embed_train_industry_id_seq.pkl')
		with open(save_path, 'wb') as f:
			pickle.dump(seq_dedup, f)

		del join, merge, dic, dic_dedup, train_seq, test_seq, seq_dedup
		_ = gc.collect()

	logger.info('Industry ID sequence data ready')

	# Prepare Product Category Sequence Data
	if not os.path.isfile(os.path.join(input_path, 'train_product_category_id_seq.pkl')):
		join = ad[['creative_id', 'product_category']].drop_duplicates()
		merge = pd.merge(cl, join, on='creative_id')
		merge.sort_values(['user_id', 'time'], inplace=True)
		assert merge.shape[0]==cl.shape[0]

		dic = {}
		dic_dedup = {}

		for user, i in tqdm.tqdm(merge[['user_id', 'product_category']].values, desc='product category id'):
			i = str(i)
			if user in dic:
				dic[user].append(i)
				if i not in dic_dedup[user]:
					dic_dedup[user].append(i)
			else:
				dic[user] = [i]
				dic_dedup[user] = [i]

		train_seq = []
		test_seq = []
		seq_dedup = []

		for user in train_idx:
			train_seq.append(dic[user])
			seq_dedup.append(dic_dedup[user])

		for user in test_idx:
			test_seq.append(dic[user])
			seq_dedup.append(dic_dedup[user])

		save_path = os.path.join(input_path, 'train_product_category_id_seq.pkl')
		with open(save_path, 'wb') as f:
			pickle.dump(train_seq, f)

		save_path = os.path.join(input_path, 'test_product_category_id_seq.pkl')
		with open(save_path, 'wb') as f:
			pickle.dump(test_seq, f)

		save_path = os.path.join(embed_path, 'embed_train_product_category_id_seq.pkl')
		with open(save_path, 'wb') as f:
			pickle.dump(seq_dedup, f)

		del join, merge, dic, dic_dedup, train_seq, test_seq, seq_dedup
		_ = gc.collect()

	logger.info('Product Category ID sequence data ready')

	del cl, ad
	_ = gc.collect()


if __name__=='__main__':
	logger = initiate_logger('input_generate.log')
	generate(logger=logger)