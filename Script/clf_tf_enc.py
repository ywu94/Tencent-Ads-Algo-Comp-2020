"""
Reference:
[1] https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, kaiming_normal_
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class Positional_Encoding_Layer(nn.Module):
	"""
	Positional encoding using sine and cosine as described in "Attention is all you need".
	"""
	def __init__(self, d_model, max_seq_len=512, dropout=0.2):
		"""
		Formula:
		| PE(pos,2i) = sin(pos/10000**(2*i/d_model))
		| PE(pos,2i+1) = cos(pos/10000**(2*i/d_model))
		"""
		super(Positional_Encoding_Layer, self).__init__()
		self.d_model = d_model
		self.dropout = dropout
		self.max_seq_len = max_seq_len

		self.dropout_layer = nn.Dropout(p=dropout)
		pe = torch.zeros(max_seq_len, d_model)                                                       # (max_seq_len, d_model)
		position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)                      # (max_seq_len, 1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))     # (d_model/2)
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term) 
		pe = pe.unsqueeze(0).transpose(0, 1)                                                         # (max_seq_len, 1, d_model)
		self.register_buffer('pe', pe)

	def forward(self, inp):
		inp = inp + self.pe[:inp.size(0), :]                                                         # (n_step, batch_size, d_model)
		return self.dropout_layer(inp)

class Transformer_Encoder_Extraction_Layer(nn.Module):
	"""
	Transformer encoder as described in "Attention is all you need", followed by a single LSTM layer.
	"""
	def __init__(self, n_enc_layer, embed_size, n_head, intermediate_size, use_pe=True, max_seq_len=512, dropout=0.2, **kwargs):
		super(Transformer_Encoder_Extraction_Layer, self).__init__(**kwargs)
		assert embed_size%n_head==0

		self.n_enc_layer = n_enc_layer
		self.embed_size = embed_size
		self.max_seq_len = max_seq_len
		self.n_head = n_head
		self.intermediate_size = intermediate_size
		self.use_pe = use_pe
		self.dropout = dropout

		if self.use_pe: 
			self.positional_encoder = Positional_Encoding_Layer(embed_size, max_seq_len=max_seq_len)
		transformer_encoder_layer = TransformerEncoderLayer(embed_size, n_head, dim_feedforward=intermediate_size, dropout=dropout)
		self.transformer_encoder = TransformerEncoder(transformer_encoder_layer, n_enc_layer)
		self.lstm_layer = nn.LSTM(input_size=embed_size, hidden_size=embed_size, bias=True)

		self._init_weights()

	def _init_weights(self):
		for p in self.parameters():
			if p.dim() > 1:
				kaiming_normal_(p, mode='fan_out', nonlinearity='relu')

	def forward(self, inp, inp_padding_mask=None):
		inp = (inp * np.sqrt(self.embed_size)).permute(1, 0, 2)                        # (batch_size, n_step, embed_size)
		if self.use_pe:
			inp = self.positional_encoder(inp)                                         # (n_step, batch_size, embed_size)
		out = self.transformer_encoder(inp, src_key_padding_mask=inp_padding_mask)     # (n_step, batch_size, embed_size)
		out, _ = self.lstm_layer(out)                                                  # (n_step, batch_size, embed_size)
		return out.permute(1, 0, 2)

class MLP_Classification_Layer(nn.Module):
	"""
	Multilayer Perception Classification Layer
	- Layer 1: Linear + Batchnorm + ReLU + Dropout
	- Layer 2: Linear + Batchnorm + ReLU + Dropout
	- Layer 3: Linear
	"""
	def __init__(self, inp_size, out_size, dropout=0.5, **kwargs):
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
	
	def _init_weights(self):
		initrange = 0.1
		self.mlp_1.weight.data.uniform_(-initrange, initrange)
		self.mlp_1.bias.data.zero_()
		self.mlp_2.weight.data.uniform_(-initrange, initrange)
		self.mlp_2.bias.data.zero_()
		self.mlp_3.weight.data.uniform_(-initrange, initrange)
		self.mlp_3.bias.data.zero_()

	def forward(self, inp):
		mlp_out = self.mlp_1(inp)                                                         # (batch_size, 1024)
		mlp_out = self.mlp_dropout_1(F.relu(self.batchnorm_1(mlp_out)))                   # (batch_size, 1024)
		mlp_out = self.mlp_2(mlp_out)                                                     # (batch_size, 512)
		mlp_out = self.mlp_dropout_2(F.relu(self.batchnorm_2(mlp_out)))                   # (batch_size, 512)
		mlp_out = self.mlp_3(mlp_out)                                                     # (batch_size, out_size)
		return mlp_out   
	
class Transformer_Encoder_Classifier(nn.Module):
	"""
	Transformer Encoder + LSTM + Multilayer Perception for Classification
	"""
	def __init__(self, embed_size, out_size, n_enc_layer, n_head, intermediate_size, device=None, max_seq_len=100, rnn_dropout=0.2, dnn_dropout=0.5, **kwargs):
		super(Transformer_Encoder_Classifier, self).__init__(**kwargs)
		
		self.embed_size = embed_size
		self.out_size = out_size
		self.n_enc_layer = n_enc_layer
		self.n_head = n_head
		self.intermediate_size = intermediate_size
		self.device = device if device else torch.device('cpu')
		self.max_seq_len = max_seq_len
		self.rnn_dropout = rnn_dropout
		self.dnn_dropout = dnn_dropout

		self.encoder_layer = Transformer_Encoder_Extraction_Layer(n_enc_layer, embed_size, n_head, intermediate_size, use_pe=False, dropout=rnn_dropout)
		self.max_pooling = nn.MaxPool1d(max_seq_len)
		self.inp_dropout = nn.Dropout(p=dnn_dropout)
		self.mlp_layer = MLP_Classification_Layer(embed_size, out_size, dropout=dnn_dropout)

	def get_padding_mask(self, batch_size, seq_len, inp_len):
		padding_mask = np.ones((batch_size, seq_len))
		for index, l in enumerate(inp_len):
			padding_mask[index,:l] = 0
		return torch.from_numpy(padding_mask).bool().to(self.device)

	def forward(self, inp_embed, inp_len):
		batch_size, seq_len, _ = inp_embed.shape
		inp_padding_mask = self.get_padding_mask(batch_size, seq_len, inp_len)
		out = self.encoder_layer(inp_embed, inp_padding_mask=inp_padding_mask)               # (batch_size, n_step, embed_size)
		out = self.max_pooling(out.permute(0,2,1)).squeeze(2)                                # (batch_size, embed_size)
		out = self.mlp_layer(self.inp_dropout(out))                                          # (batch_size, out_size)
		return out

class Multi_Seq_Transformer_Encoder_Classifier(nn.Module):
	"""
	Transformer Encoder + LSTM + Multilayer Perception for Classification, taks multiple sequences as input
	"""
	def __init__(self, embed_size, out_size, n_enc_layer, n_head, intermediate_size, device=None, max_seq_len=100, rnn_dropout=0.2, dnn_dropout=0.5, **kwargs):
		super(Multi_Seq_Transformer_Encoder_Classifier, self).__init__(**kwargs)
		
		self.embed_size = embed_size
		self.out_size = out_size
		self.n_enc_layer = n_enc_layer
		self.n_head = n_head
		self.intermediate_size = intermediate_size
		self.device = device if device else torch.device('cpu')
		self.max_seq_len = max_seq_len
		self.rnn_dropout = rnn_dropout
		self.dnn_dropout = dnn_dropout

		self.n_extraction = len(embed_size)
		self.mlp_inp_size = sum(embed_size)

		for index, e_size in enumerate(embed_size):
			setattr(self, 'encoder_layer_{}'.format(index), Transformer_Encoder_Extraction_Layer(n_enc_layer, e_size, n_head, intermediate_size, use_pe=False, dropout=rnn_dropout))
		self.max_pooling = nn.MaxPool1d(max_seq_len)
		self.inp_dropout = nn.Dropout(p=dnn_dropout)
		self.mlp_layer = MLP_Classification_Layer(self.mlp_inp_size, out_size, dropout=dnn_dropout)

	def get_padding_mask(self, batch_size, seq_len, inp_len):
		padding_mask = np.ones((batch_size, seq_len))
		for index, l in enumerate(inp_len):
			padding_mask[index,:l] = 0
		return torch.from_numpy(padding_mask).bool().to(self.device)

	def forward(self, *args):
		assert len(args)==self.n_extraction+1
		batch_size, seq_len, _ = args[0].shape
		inp_padding_mask = self.get_padding_mask(batch_size, seq_len, args[-1])
		ext_buf = [getattr(self, 'encoder_layer_{}'.format(index))(i, inp_padding_mask=inp_padding_mask) for index, i in enumerate(args[:-1])]
		out = torch.cat([self.max_pooling(i.permute(0,2,1)).squeeze(2) for i in ext_buf], dim=1)
		out = self.mlp_layer(self.inp_dropout(out))                                          # (batch_size, out_size)
		return out