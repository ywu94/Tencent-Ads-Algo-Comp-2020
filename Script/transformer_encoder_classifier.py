"""
Reference:
[1] https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class Positional_Encoding_Layer(nn.Module):
	"""
	Positional encoding using sine and cosine as described in "Attention is all you need".
	"""
	def __init__(self, d_model, max_seq_len=512, dropout=0.1):
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
	Transformer encoder as described in "Attention is all you need".
	"""
	def __init__(self, n_enc_layer, embed_size, n_head, intermediate_size, max_seq_len=512, dropout=0.1, **kwargs):
		super(Transformer_Encoder_Extraction_Layer, self).__init__(**kwargs)
		assert embed_size%n_head==0

		self.n_enc_layer = n_enc_layer
		self.embed_size = embed_size
		self.max_seq_len = max_seq_len
		self.n_head = n_head
		self.intermediate_size = intermediate_size
		self.dropout = dropout

		self.positional_encoder = Positional_Encoding_Layer(embed_size, max_seq_len=max_seq_len)
		transformer_encoder_layer = TransformerEncoderLayer(embed_size, n_head, dim_feedforward=intermediate_size, dropout=dropout)
		self.transformer_encoder = TransformerEncoder(transformer_encoder_layer, n_enc_layer)

		self._init_weights()

	def _init_weights(self):
		for p in self.parameters():
			if p.dim() > 1:
				xavier_uniform_(p)

	def forward(self, inp, inp_padding_mask=None):
		inp = inp * np.sqrt(self.embed_size)                                           # (batch_size, n_step, embed_size)
		inp = self.positional_encoder(inp.permute(1, 0, 2))                            # (n_step, batch_size, embed_size)
		out = self.transformer_encoder(inp, src_key_padding_mask=inp_padding_mask)     # (n_step, batch_size, embed_size)
		return out.permute(1, 0, 2)

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
	Transformer Encoder + Multilayer Perception for Classification
	"""
	def __init__(self, embed_size, out_size, n_enc_layer, n_head, intermediate_size, device, transformer_dropout=0.1, mlp_dropout=0.4, **kwargs):
		super(Transformer_Encoder_Classifier, self).__init__(**kwargs)
		
		self.embed_size = embed_size
		self.out_size = out_size
		self.n_enc_layer = n_enc_layer
		self.n_head = n_head
		self.intermediate_size = intermediate_size
		self.device = device
		self.transformer_dropout = transformer_dropout
		self.mlp_dropout = mlp_dropout

		self.encoder_layer = Transformer_Encoder_Extraction_Layer(n_enc_layer, embed_size, n_head, intermediate_size, dropout=transformer_dropout)
		self.classification_layer = MLP_Classification_Layer(embed_size, out_size, dropout=mlp_dropout)

	def get_padding_mask(self, batch_size, seq_len, inp_last_idx):
		padding_mask = np.ones((batch_size, seq_len))
		for index, last_idx in enumerate(inp_last_idx):
			padding_mask[index,:last_idx+1] = 0
		return torch.from_numpy(padding_mask).bool().to(self.device)

	def forward(self, inp_embed, inp_last_idx):
		assert inp_embed.shape[0] == inp_last_idx.shape[0]
		batch_size = inp_embed.shape[0]
		seq_len = inp_embed.shape[1]
		inp_padding_mask = self.get_padding_mask(batch_size, seq_len, inp_last_idx)
		out = self.encoder_layer(inp_embed, inp_padding_mask=inp_padding_mask)               # (batch_size, n_step, embed_size)
		pooled_buf = []
		for index, last_idx in enumerate(inp_last_idx):
			pooled_buf.append(torch.mean(out[index,:last_idx+1,:], dim=0))
		out = torch.stack(pooled_buf)                                                        # (batch_size, embed_size)
		out = self.classification_layer(out)                                                 # (batch_size, out_size)
		return out
