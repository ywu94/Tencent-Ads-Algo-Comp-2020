import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

class LSTM_Extraction_Layer(nn.Module):
	"""
	LSTM feature extration layer
	- Layer 1: BiLSTM + Dropout + Layernorm
	- Layer 2: LSTM with Residual Connection + Dropout + Layernorm
	- Layer 3: LSTM + Batchnorm + ReLU + Dropout
	"""
	def __init__(self, embed_size, lstm_hidden_size, rnn_dropout=0.2, mlp_dropout=0.4, **kwargs):
		super(LSTM_Extraction_Layer, self).__init__(**kwargs)
		self.embed_size = embed_size
		self.lstm_hidden_size = lstm_hidden_size
		self.rnn_dropout = rnn_dropout
		self.mlp_dropout = mlp_dropout
		
		self.bi_lstm = nn.LSTM(input_size=embed_size, hidden_size=lstm_hidden_size, bias=True, bidirectional=True)
		self.rnn_dropout_1 = nn.Dropout(p=rnn_dropout)
		self.layernorm_1 = nn.LayerNorm(2*lstm_hidden_size)
		self.lstm_1 = nn.LSTM(input_size=2*lstm_hidden_size, hidden_size=2*lstm_hidden_size)
		self.rnn_dropout_2 = nn.Dropout(p=rnn_dropout)
		self.layernorm_2 = nn.LayerNorm(2*lstm_hidden_size)
		self.lstm_2 = nn.LSTM(input_size=2*lstm_hidden_size, hidden_size=2*lstm_hidden_size)
		self.batchnorm = nn.BatchNorm1d(2*lstm_hidden_size)
		self.mlp_dropout = nn.Dropout(p=mlp_dropout)
		
	def forward(self, inp_embed, inp_last_idx):
		bilstm_out, _ = self.bi_lstm(inp_embed.permute(1,0,2))                            # (max_seq_length, batch_size, embed_size) -> (max_seq_length, batch_size, 2*lstm_hidden_size)
		bilstm_out = self.layernorm_1(self.rnn_dropout_1(bilstm_out))                     # (max_seq_length, batch_size, 2*lstm_hidden_size)
		lstm_out, _ = self.lstm_1(bilstm_out)                                             # (max_seq_length, batch_size, 2*lstm_hidden_size)
		lstm_out = self.rnn_dropout_2(lstm_out)                                           # (max_seq_length, batch_size, 2*lstm_hidden_size)
		lstm_out = self.layernorm_2(lstm_out+bilstm_out)                                  # (max_seq_length, batch_size, 2*lstm_hidden_size)
		lstm_out, _ = self.lstm_2(lstm_out)                                               # (max_seq_length, batch_size, 2*lstm_hidden_size)
		lstm_out = lstm_out.permute(1,0,2)[np.arange(len(inp_last_idx)), inp_last_idx,:]  # (batch_size, 2*lstm_hidden_size)
		lstm_out = self.mlp_dropout(F.relu(self.batchnorm(lstm_out)))                     # (batch_size, 2*lstm_hidden_size)
		return lstm_out
	
class MLP_Classification_Layer(nn.Module):
	"""
	Multilayer Perception Classification Layer
	- Layer 1: Linear + Batchnorm + ReLU + Dropout
	- Layer 2: Linear + Batchnorm + ReLU + Dropout
	- Layer 3: Linear
	"""
	def __init__(self, inp_size, out_size, dropout=0.4, **kwargs):
		super(MLP_Classification_Layer, self).__init__(**kwargs)
		self.inp_size = inp_size
		self.out_size = out_size
		self.dropout = dropout
		
		self.mlp_1 = nn.Linear(inp_size, 1024)
		self.batchnorm_1 = nn.BatchNorm1d(1024)
		self.mlp_dropout_1 = nn.Dropout(p=dropout)
		self.mlp_2 = nn.Linear(1024, 512)
		self.batchnorm_2 = nn.BatchNorm1d(512)
		self.mlp_dropout_2 = nn.Dropout(p=dropout)
		self.mlp_3 = nn.Linear(512, out_size)
		
	def forward(self, inp):
		mlp_out = self.mlp_1(inp)                                                         # (batch_size, 1024)
		mlp_out = self.mlp_dropout_1(F.relu(self.batchnorm_1(mlp_out)))                   # (batch_size, 1024)
		mlp_out = self.mlp_2(mlp_out)                                                     # (batch_size, 512)
		mlp_out = self.mlp_dropout_2(F.relu(self.batchnorm_2(mlp_out)))                   # (batch_size, 512)
		mlp_out = self.mlp_3(mlp_out)                                                     # (batch_size, out_size)
		return mlp_out   
	
class Multi_Seq_LSTM_Classifier(nn.Module):
	"""
	Use separate LSTM extractor to handle different sequences, concat them and feed backto multilayer perception classifier.
	"""
	def __init__(self, embed_size, lstm_hidden_size, out_size, rnn_dropout=0.2, mlp_dropout=0.4, **kwargs):
		super(Multi_Seq_LSTM_Classifier, self).__init__(**kwargs)
		assert isinstance(embed_size, list) and isinstance(lstm_hidden_size, list) and len(embed_size)==len(lstm_hidden_size)
		
		self.embed_size = embed_size
		self.lstm_hidden_size = lstm_hidden_size
		self.out_size = out_size
		self.rnn_dropout = rnn_dropout
		self.mlp_dropout = mlp_dropout
		
		self.n_extraction = len(embed_size)
		self.mlp_inp_size = sum(map(lambda x:2*x, lstm_hidden_size))
		
		for index, (e_size, h_size) in enumerate(zip(embed_size, lstm_hidden_size)):
			setattr(self, f'extraction_layer_{index}', LSTM_Extraction_Layer(e_size, h_size, rnn_dropout=rnn_dropout, mlp_dropout=mlp_dropout))
		self.classification_layer = MLP_Classification_Layer(self.mlp_inp_size, out_size, dropout=mlp_dropout)
		
	def forward(self, *args):
		assert len(args)==self.n_extraction+1
		
		extract_buffer = [getattr(self, f'extraction_layer_{index}')(inp_embed, args[-1]) for index, inp_embed in enumerate(args[:-1])]
		out = torch.cat(extract_buffer, 1)
		out = self.classification_layer(out)
		
		return out