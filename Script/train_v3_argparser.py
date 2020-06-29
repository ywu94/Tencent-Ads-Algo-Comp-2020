import sys
import time
import argparse
import logging

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

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('model', help='model to train')
	parser.add_argument('-ep', '--epoch', type=int, default=40, help='number of epoches to train')
	parser.add_argument('-bs', '--batchsize', type=int, default=1024, help='size of mini batch')
	parser.add_argument('-sl', '--seqlen', type=int, default=100, help='max sequence length')
	parser.add_argument('-lr', '--learningrate', type=float, default=1e-3, help='learning rate')	
	group = parser.add_mutually_exclusive_group()
	group.add_argument('-st', '--start', action='store_true', default=True, help='start training from scractch')
	group.add_argument('-re', '--resume', type=str, help='resume training from checkpoint')
	args = parser.parse_args()

	logger = initiate_logger('train_v3.log')
	logger.info('Model: {} - {}\nEpoch: {}, Batch Size: {}, Sequence Length: {}, Learning Rate: {}'.format(
		args.model, 
		'Start training from scratch' if not args.resume else 'Resume training from checkpoint {}'.format(args.resume),
		args.epoch, args.batchsize, args.seqlen, args.learningrate
	))