import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

# ========================== #
#      Final Model - 1       #
# ========================== #

class Extraction_LSTM(nn.Module):
	def __init__(self, embed_size, hidden_size, max_seq_len=100, dropout=0.2, **kwargs):
		super(Extraction_LSTM, self).__init__(**kwargs)
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.max_seq_len = max_seq_len
		self.dropout = dropout
		
		self.bi_lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
		self.dropout_1 = nn.Dropout(p=dropout)
		self.layernorm_1 = nn.LayerNorm(2*hidden_size)
		self.lstm_1 = nn.LSTM(input_size=2*hidden_size, hidden_size=2*hidden_size, batch_first=True)
		self.dropout_2 = nn.Dropout(p=dropout)
		self.layernorm_2 = nn.LayerNorm(2*hidden_size)
		self.lstm_2 = nn.LSTM(input_size=2*hidden_size, hidden_size=2*hidden_size, batch_first=True)
		
	def forward(self, inp, inp_len):
		inp = nn.utils.rnn.pack_padded_sequence(inp, batch_first=True, lengths=inp_len, enforce_sorted=False)
		inp = self.bi_lstm(inp)[0]
		inp = self.dropout_1(nn.utils.rnn.pad_packed_sequence(inp, batch_first=True, total_length=self.max_seq_len)[0])
		inp1 = nn.utils.rnn.pack_padded_sequence(self.layernorm_1(inp), batch_first=True, lengths=inp_len, enforce_sorted=False)
		inp1 = self.lstm_1(inp1)[0]
		inp = inp + self.dropout_2(nn.utils.rnn.pad_packed_sequence(inp1, batch_first=True, total_length=self.max_seq_len)[0])
		inp = nn.utils.rnn.pack_padded_sequence(self.layernorm_2(inp), batch_first=True, lengths=inp_len, enforce_sorted=False)
		inp = self.lstm_2(inp)[0]
		inp = nn.utils.rnn.pad_packed_sequence(inp, batch_first=True, total_length=self.max_seq_len)[0]
		return inp

class Output_MLP(nn.Module):
	def __init__(self, inp_size, out_size, dropout=0.5, **kwargs):
		super(Output_MLP, self).__init__(**kwargs)
		self.inp_size = inp_size
		self.out_size = out_size
		self.dropout = dropout
		
		self.mlp_1 = nn.Linear(inp_size, 4096)
		self.batchnorm_1 = nn.BatchNorm1d(4096)
		self.mlp_dropout_1 = nn.Dropout(p=dropout)
		self.mlp_2 = nn.Linear(4096, 2048)
		self.batchnorm_2 = nn.BatchNorm1d(2048)
		self.mlp_dropout_2 = nn.Dropout(p=dropout)
		self.mlp_3 = nn.Linear(2048, out_size)	

	def forward(self, inp):
		mlp_out = self.mlp_1(inp)                                                         # (batch_size, 1024)
		mlp_out = self.mlp_dropout_1(F.relu(self.batchnorm_1(mlp_out)))                   # (batch_size, 1024)
		mlp_out = self.mlp_2(mlp_out)                                                     # (batch_size, 512)
		mlp_out = self.mlp_dropout_2(F.relu(self.batchnorm_2(mlp_out)))                   # (batch_size, 512)
		mlp_out = self.mlp_3(mlp_out)                                                     # (batch_size, out_size)
		return mlp_out 

class Final_LSTM(nn.Module):
	def __init__(self, out_size, embed_size, hidden_size, max_seq_len=100, rnn_dropout=0.2, dnn_dropout=0.5, **kwargs):
		super(Final_LSTM, self).__init__(**kwargs)
		self.out_size = out_size
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.max_seq_len = max_seq_len
		self.rnn_dropout = rnn_dropout
		self.dnn_dropout = dnn_dropout

		self.extraction_layer = Extraction_LSTM(embed_size, hidden_size, max_seq_len=max_seq_len, dropout=rnn_dropout)
		self.bn_layer = nn.BatchNorm1d(4*hidden_size)
		self.dropout_layer = nn.Dropout(p=dnn_dropout)
		self.output_layer = Output_MLP(4*hidden_size, out_size, dropout=dnn_dropout)

	def forward(self, inp, inp_len):
		inp = self.extraction_layer(inp, inp_len)
		out1 = inp[np.arange(len(inp_len)),inp_len-1,:]
		out2 = torch.stack([torch.max(inp[index,:l,:], dim=0)[0] for index, l in enumerate(inp_len)], dim=0)
		out = self.dropout_layer(F.relu(self.bn_layer(torch.cat((out1, out2), dim=1))))
		out = self.output_layer(out)
		return out