import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

# ========================== #
#       Shared Layer         #
# ========================== #

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
		mlp_out = self.mlp_1(inp)                                                         # (batch_size, 4096)
		mlp_out = self.mlp_dropout_1(F.relu(self.batchnorm_1(mlp_out)))                   # (batch_size, 4096)
		mlp_out = self.mlp_2(mlp_out)                                                     # (batch_size, 2048)
		mlp_out = self.mlp_dropout_2(F.relu(self.batchnorm_2(mlp_out)))                   # (batch_size, 2048)
		mlp_out = self.mlp_3(mlp_out)                                                     # (batch_size, out_size)
		return mlp_out 

class Output_DMLP(nn.Module):
	def __init__(self, inp_size, out_size, dropout=0.5, **kwargs):
		super(Output_DMLP, self).__init__(**kwargs)
		self.inp_size = inp_size
		self.out_size = out_size
		self.dropout = dropout
		
		self.mlp_1 = nn.Linear(inp_size, inp_size*4)
		self.batchnorm_1 = nn.BatchNorm1d(inp_size*4)
		self.mlp_dropout_1 = nn.Dropout(p=dropout)
		self.mlp_2 = nn.Linear(inp_size*4, inp_size)
		self.batchnorm_2 = nn.BatchNorm1d(inp_size)
		self.mlp_dropout_2 = nn.Dropout(p=dropout)
		self.mlp_3 = nn.Linear(inp_size, out_size)	

	def forward(self, inp):
		mlp_out = self.mlp_1(inp)                                                         # (batch_size, 4*inp_size)
		mlp_out = self.mlp_dropout_1(F.relu(self.batchnorm_1(mlp_out)))                   # (batch_size, 4*inp_size)
		mlp_out = self.mlp_2(mlp_out)                                                     # (batch_size, inp_size)
		mlp_out = self.mlp_dropout_2(F.relu(self.batchnorm_2(mlp_out)))                   # (batch_size, inp_size)
		mlp_out = self.mlp_3(mlp_out)                                                     # (batch_size, out_size)
		return mlp_out 

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

# ========================== #
#      Final Model - 2       #
# ========================== #

class Extraction_GRU(nn.Module):
	def __init__(self, embed_size, hidden_size, max_seq_len=100, dropout=0.2, **kwargs):
		super(Extraction_GRU, self).__init__(**kwargs)
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.max_seq_len = max_seq_len
		self.dropout = dropout
		
		self.bi_gru = nn.GRU(input_size=embed_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
		self.dropout_1 = nn.Dropout(p=dropout)
		self.layernorm_1 = nn.LayerNorm(2*hidden_size)
		self.gru_1 = nn.GRU(input_size=2*hidden_size, hidden_size=2*hidden_size, batch_first=True)
		self.dropout_2 = nn.Dropout(p=dropout)
		self.layernorm_2 = nn.LayerNorm(2*hidden_size)
		self.gru_2 = nn.GRU(input_size=2*hidden_size, hidden_size=2*hidden_size, batch_first=True)
		
	def forward(self, inp, inp_len):
		inp = self.dropout_1(self.bi_gru(inp)[0])
		inp1 = self.layernorm_1(inp)
		inp1 = self.dropout_2(self.gru_1(inp1)[0])
		inp = self.layernorm_2(inp+inp1)
		inp = self.gru_2(inp)[0]
		return inp

class Final_GRU(nn.Module):
	def __init__(self, out_size, embed_size, hidden_size, max_seq_len=100, rnn_dropout=0.2, dnn_dropout=0.5, **kwargs):
		super(Final_GRU, self).__init__(**kwargs)
		self.out_size = out_size
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.max_seq_len = max_seq_len
		self.rnn_dropout = rnn_dropout
		self.dnn_dropout = dnn_dropout

		self.extraction_layer = Extraction_GRU(embed_size, hidden_size, max_seq_len=max_seq_len, dropout=rnn_dropout)
		self.bn_layer = nn.BatchNorm1d(4*hidden_size)
		self.dropout_layer = nn.Dropout(p=dnn_dropout)
		self.output_layer = Output_DMLP(4*hidden_size, out_size, dropout=dnn_dropout)

	def forward(self, inp, inp_len):
		inp = self.extraction_layer(inp, inp_len)
		out1 = inp[np.arange(len(inp_len)),inp_len-1,:]
		out2 = torch.stack([torch.max(inp[index,:l,:], dim=0)[0] for index, l in enumerate(inp_len)], dim=0)
		out = self.dropout_layer(F.relu(self.bn_layer(torch.cat((out1, out2), dim=1))))
		out = self.output_layer(out)
		return out

# ========================== #
#      Final Model - 3       #
# ========================== #

class PreLN_Transformer_Encoder(nn.Module):
	def __init__(self, d_model, n_head, intermediate_size=2048, device=None, dropout=0.1, **kwargs):
		super(PreLN_Transformer_Encoder, self).__init__(**kwargs)
		self.d_model = d_model
		self.n_head = n_head
		self.intermediate_size = intermediate_size
		self.device = device if device else torch.device('cpu')
		self.dropout = dropout

		self.ln_layer_1 = nn.LayerNorm(d_model)
		self.mha_layer = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
		self.attn_dropout = nn.Dropout(p=dropout)
		self.ln_layer_2 = nn.LayerNorm(d_model)
		self.ffn_layer_1 = nn.Linear(d_model, intermediate_size)
		self.dropout_1 = nn.Dropout(p=dropout)
		self.ffn_layer_2 = nn.Linear(intermediate_size, d_model)
		self.dropout_2 = nn.Dropout(p=dropout)

	def _get_padding_mask(self, batch_size, seq_len, inp_len):
		padding_mask = np.ones((batch_size, seq_len))
		for index, l in enumerate(inp_len):
			padding_mask[index,:l] = 0
		return torch.from_numpy(padding_mask).bool().to(self.device)

	def forward(self, inp, inp_len):
		batch_size, seq_len, _ = inp.shape
		padding_mask = self._get_padding_mask(batch_size, seq_len, inp_len)                            
		inp1 = self.ln_layer_1(inp).permute(1,0,2)                                                     
		inp2 = self.mha_layer(inp1, inp1, inp1, key_padding_mask=padding_mask)[0].permute(1,0,2)     
		inp = inp + self.attn_dropout(inp2)
		inp1 = self.ln_layer_2(inp)
		inp2 = self.ffn_layer_2(self.dropout_1(F.relu(self.ffn_layer_1(inp1))))
		inp = inp + self.dropout_2(inp2)
		return inp

class Extraction_PreLN_Transformer(nn.Module):
	def __init__(self, n_layer, d_model, n_head, intermediate_size=2048, device=None, dropout=0.1, **kwargs):
		super(Extraction_PreLN_Transformer, self).__init__(**kwargs)
		self.n_layer = n_layer
		self.d_model = d_model
		self.n_head = n_head
		self.intermediate_size = intermediate_size
		self.device = device if device else torch.device('cpu')
		self.dropout = dropout

		for index in range(n_layer):
			setattr(self, 'pre_ln_tf_encoder_{}'.format(index), 
				PreLN_Transformer_Encoder(d_model, n_head, intermediate_size=intermediate_size, device=self.device, dropout=dropout))

	def forward(self, inp, inp_len):
		for index in range(self.n_layer):
			inp = getattr(self, 'pre_ln_tf_encoder_{}'.format(index))(inp, inp_len)
		return inp

class Final_PreLN_Transformer(nn.Module):
	def __init__(self, out_size, embed_size, n_layer, n_head, intermediate_size=2048, device=None, max_seq_len=100, tf_dropout=0.1, rnn_dropout=0.2, dnn_dropout=0.5, **kwargs):
		super(Final_PreLN_Transformer, self).__init__(**kwargs)
		self.out_size = out_size
		self.embed_size = embed_size
		self.n_layer = n_layer
		self.n_head = n_head
		self.intermediate_size = intermediate_size
		self.device = device if device else torch.device('cpu')
		self.max_seq_len = max_seq_len
		self.tf_dropout = tf_dropout
		self.rnn_dropout = rnn_dropout
		self.dnn_dropout = dnn_dropout

		self.extraction_layer = Extraction_PreLN_Transformer(n_layer, embed_size, n_head, dropout=tf_dropout, device=self.device)
		self.ln_layer = nn.LayerNorm(embed_size)
		self.lstm_layer = nn.LSTM(input_size=embed_size, hidden_size=embed_size, batch_first=True)
		self.dropout1 = nn.Dropout(p=rnn_dropout)
		self.bn_layer = nn.BatchNorm1d(2*embed_size)
		self.dropout2 = nn.Dropout(p=dnn_dropout)
		self.output_layer = Output_MLP(2*embed_size, out_size, dropout=dnn_dropout)

	def forward(self, inp, inp_len):
		inp = self.extraction_layer(inp, inp_len)
		inp1 = self.lstm_layer(self.ln_layer(inp))[0]
		inp = inp + self.dropout1(inp1)
		out1 = inp[np.arange(len(inp_len)),inp_len-1,:]
		out2 = torch.stack([torch.max(inp[index,:l,:], dim=0)[0] for index, l in enumerate(inp_len)], dim=0)
		out = self.dropout2(F.relu(self.bn_layer(torch.cat((out1, out2), dim=1))))
		out = self.output_layer(out)
		return out
