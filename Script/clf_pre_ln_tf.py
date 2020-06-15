import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, kaiming_normal_

class Pre_LN_Transformer_Encoder_Layer(nn.Module):
	"""
	Encoder layer for Pre-LN Transformer
	"""
	def __init__(self, d_model, n_head, intermediate_size=2048, device=None, dropout=0.1, **kwargs):
		super(Pre_LN_Transformer_Encoder_Layer, self).__init__(**kwargs)
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
		padding_mask = self._get_padding_mask(batch_size, seq_len, inp_len)                            # (batch_size, seq_len)
		inp1 = self.ln_layer_1(inp).permute(1,0,2)                                                     # (seq_len, batch_size, d_model)
		inp2 = self.mha_layer(inp1, inp1, inp1, key_padding_mask=padding_mask)[0].permute(1,0,2)       # (batch_size, seq_len, d_model)
		inp = inp + self.attn_dropout(inp2)
		inp1 = self.ln_layer_2(inp)
		inp2 = self.ffn_layer_2(self.dropout_1(F.relu(self.ffn_layer_1(inp1))))
		inp = inp + self.dropout_2(inp2)
		return inp

class Pre_LN_Transformer_Encoder(nn.Module):
	"""
	Stacked Pre-LN Transformer Encoder layers
	"""
	def __init__(self, n_layer, d_model, n_head, intermediate_size=2048, device=None, dropout=0.1, **kwargs):
		super(Pre_LN_Transformer_Encoder, self).__init__(**kwargs)
		self.n_layer = n_layer
		self.d_model = d_model
		self.n_head = n_head
		self.intermediate_size = intermediate_size
		self.device = device if device else torch.device('cpu')
		self.dropout = dropout

		for index in range(n_layer):
			setattr(self, 'pre_ln_tf_encoder_{}'.format(index), 
				Pre_LN_Transformer_Encoder_Layer(d_model, n_head, intermediate_size=intermediate_size, device=self.device, dropout=0.1))

	def forward(self, inp, inp_len):
		for index in range(self.n_layer):
			inp = getattr(self, 'pre_ln_tf_encoder_{}'.format(index))(inp, inp_len)
		return inp

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

class Multi_Seq_Pre_LN_Transformer_Encoder_Classifier(nn.Module):
	def __init__(self, embed_size, hidden_size, n_layer, n_head, out_size, intermediate_size=2048, max_seq_len=100, device=None, tf_dropout=0.1, rnn_dropout=0.2, dnn_dropout=0.5, **kwargs):
		super(Multi_Seq_Pre_LN_Transformer_Encoder_Classifier, self).__init__(**kwargs)
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.n_layer = n_layer
		self.n_head = n_head
		self.out_size = out_size
		self.intermediate_size = intermediate_size
		self.max_seq_len = max_seq_len
		self.device = device if device else torch.device('cpu')
		self.tf_dropout = tf_dropout
		self.rnn_dropout = rnn_dropout
		self.dnn_dropout = dnn_dropout

		self.n_extraction = len(embed_size)
		self.mlp_inp_size = sum(map(lambda x:4*x, hidden_size))

		for index, e_size in enumerate(embed_size):
			setattr(self, 'pre_ln_tf_encoder_{}'.format(index), Pre_LN_Transformer_Encoder(n_layer, e_size, n_head, intermediate_size=intermediate_size, device=self.device, dropout=tf_dropout))
			setattr(self, 'ln_{}'.format(index), nn.LayerNorm(e_size))

		for index, (e_size, h_size) in enumerate(zip(embed_size, hidden_size)):
			setattr(self, 'lstm_{}'.format(index), nn.LSTM(input_size=e_size, hidden_size=h_size, bias=True, bidirectional=True))

		self.max_pooling = nn.MaxPool1d(max_seq_len)

		self.inp_bn = nn.BatchNorm1d(self.mlp_inp_size)
		self.inp_dropout = nn.Dropout(p=dnn_dropout)
		self.mlp_layer = MLP_Classification_Layer(self.mlp_inp_size, out_size, dropout=dnn_dropout)

	def forward(self, *args):
		assert len(args)==self.n_extraction+1
		buf, inp_len = [], args[-1]
		for index, inp in enumerate(args[:-1]):
			inp = getattr(self, 'pre_ln_tf_encoder_{}'.format(index))(inp, inp_len)                 # (batch_size, seq_len, embed_size)
			inp = getattr(self, 'ln_{}'.format(index))(inp)                                         # (batch_size, seq_len, embed_size)
			inp = getattr(self, 'lstm_{}'.format(index))(inp.permute(1,0,2))[0].permute(1,0,2)      # (batch_size, seq_len, 2*hidden_size)
			buf.append(inp[np.arange(len(inp_len)), inp_len-1, :])                                  # (batch_size, 2*hidden_size)
			buf.append(self.max_pooling(inp.permute(0,2,1)).squeeze(2))                             # (batch_size, 2*hidden_size)
		out = self.inp_bn(torch.cat(buf, dim=1))                                                    # (batch_size, Σ4*hidden_size)
		out = self.mlp_layer(self.inp_dropout(F.relu(out)))                                         # (batch_size, out_size)
		return out





