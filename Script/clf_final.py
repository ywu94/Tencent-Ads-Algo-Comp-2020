import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

# ========================== #
#       Utility Layer        #
# ========================== #

class Kmax_Pooling_Layer(nn.Module):
	def __init__(self, dim, k, **kwargs):
		super(Kmax_Pooling_Layer, self).__init__(**kwargs)
		self.dim = dim
		self.k = k

	def forward(self, inp):
		index = inp.topk(self.k, dim=self.dim)[1].sort(dim=self.dim)[0]
		return inp.gather(self.dim, index)

# ========================== #
#       Output Layer         #
# ========================== #

class Output_MLP(nn.Module):
	def __init__(self, inp_size, out_size, dropout=0.5, **kwargs):
		super(Output_MLP, self).__init__(**kwargs)
		self.inp_size = inp_size
		self.out_size = out_size
		self.dropout = dropout

		self.l1 = nn.Linear(inp_size, 1024)
		self.l2 = nn.Linear(1024, 512)
		self.l3 = nn.Linear(512, 256)
		self.l4 = nn.Linear(256, out_size)

		self.bn1 = nn.BatchNorm1d(1024)	
		self.bn2 = nn.BatchNorm1d(512)
		self.bn3 = nn.BatchNorm1d(256)

		self.dropout1 = nn.Dropout(p=dropout)
		self.dropout2 = nn.Dropout(p=dropout)
		self.dropout3 = nn.Dropout(p=dropout)	

		self._reset_weights()
		
	def _reset_weights(self):
		nn.init.kaiming_uniform_(self.l1.weight.data, nonlinearity='leaky_relu', a=0.01)
		nn.init.zeros_(self.l1.bias.data)
		nn.init.kaiming_uniform_(self.l2.weight.data, nonlinearity='leaky_relu', a=0.01)
		nn.init.zeros_(self.l2.bias.data)
		nn.init.kaiming_uniform_(self.l3.weight.data, nonlinearity='leaky_relu', a=0.01)
		nn.init.zeros_(self.l3.bias.data)
		nn.init.kaiming_uniform_(self.l4.weight.data, nonlinearity='leaky_relu', a=0.01)
		nn.init.zeros_(self.l4.bias.data)

	def forward(self, inp):
		inp = self.dropout1(F.leaky_relu(self.bn1(self.l1(inp)), negative_slope=0.01))
		inp = self.dropout2(F.leaky_relu(self.bn2(self.l2(inp)), negative_slope=0.01))
		inp = self.dropout3(F.leaky_relu(self.bn3(self.l3(inp)), negative_slope=0.01))
		inp = self.l4(inp)
		return inp

# ========================== #
#      Extraction Layer      #
# ========================== #

class Extraction_ResGRU(nn.Module):
	def __init__(self, embed_size, hidden_size, max_seq_len=100, dropout=0.2, **kwargs):
		super(Extraction_ResGRU, self).__init__(**kwargs)
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.max_seq_len = max_seq_len
		self.dropout = dropout
		
		self.gru1 = nn.GRU(input_size=embed_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
		self.gru2 = nn.GRU(input_size=2*hidden_size, hidden_size=2*hidden_size, batch_first=True)
		self.gru3 = nn.GRU(input_size=2*hidden_size, hidden_size=2*hidden_size, batch_first=True)

		self.ln1 = nn.LayerNorm(embed_size)
		self.ln2 = nn.LayerNorm(2*hidden_size)
		self.ln3 = nn.LayerNorm(2*hidden_size)

		self.dropout1 = nn.Dropout(p=dropout)
		self.dropout2 = nn.Dropout(p=dropout)
		self.dropout3 = nn.Dropout(p=dropout)

	def _pack(self, inp, inp_len):
		return nn.utils.rnn.pack_padded_sequence(inp, batch_first=True, lengths=inp_len, enforce_sorted=False)

	def _unpack(self, inp):
		return nn.utils.rnn.pad_packed_sequence(inp, batch_first=True, total_length=self.max_seq_len)[0]

	def forward(self, inp, inp_len):
		inp = self._pack(self.dropout1(self.ln1(inp)), inp_len)
		inp = self._unpack(self.gru1(inp)[0])
		out = self._pack(self.dropout2(self.ln2(inp)), inp_len)
		inp = inp + self.dropout2(self._unpack(self.gru2(out)[0]))
		out = self._pack(self.ln3(inp), inp_len)
		inp = inp + self.dropout3(self._unpack(self.gru3(out)[0]))
		return inp

class Extraction_CNN(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
		super(Extraction_CNN, self).__init__(**kwargs)
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size

		self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
		self.bn1 = nn.BatchNorm1d(out_channels)
		self.ac1 = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)
		self.bn2 = nn.BatchNorm1d(out_channels)
		self.ac2 = nn.ReLU(inplace=True)

	def forward(self, inp):
		inp = self.ac1(self.bn1(self.conv1(inp)))                
		inp = self.ac2(self.bn2(self.conv2(inp)))
		return inp

class PreLN_Transformer_Encoder(nn.Module):
	def __init__(self, d_model, n_head, intermediate_size=2048, device=None, dropout=0.1, **kwargs):
		super(PreLN_Transformer_Encoder, self).__init__(**kwargs)
		self.d_model = d_model
		self.n_head = n_head
		self.intermediate_size = intermediate_size
		self.device = device if device else torch.device('cpu')
		self.dropout = dropout

		self.mha = nn.MultiheadAttention(d_model, n_head, dropout=dropout)

		self.ln1 = nn.LayerNorm(d_model)
		self.ln2 = nn.LayerNorm(d_model)

		self.l1 = nn.Linear(d_model, intermediate_size)
		self.l2 = nn.Linear(intermediate_size, d_model)

		self.attn_dropout = nn.Dropout(p=dropout)
		self.dropout1 = nn.Dropout(p=dropout)
		self.dropout2 = nn.Dropout(p=dropout)

		self._reset_weights()

	def _reset_weights(self):
		nn.init.kaiming_uniform_(self.l1.weight.data, nonlinearity='relu')
		nn.init.zeros_(self.l1.bias.data)
		nn.init.kaiming_uniform_(self.l2.weight.data, nonlinearity='relu')
		nn.init.zeros_(self.l2.bias.data)

	def _get_padding_mask(self, batch_size, seq_len, inp_len):
		padding_mask = np.ones((batch_size, seq_len))
		for index, l in enumerate(inp_len):
			padding_mask[index,:l] = 0
		return torch.from_numpy(padding_mask).bool().to(self.device)

	def forward(self, inp, inp_len):
		batch_size, seq_len, _ = inp.shape
		padding_mask = self._get_padding_mask(batch_size, seq_len, inp_len)  
		inp1 = self.ln1(inp).permute(1,0,2)
		inp2 = self.mha(inp1, inp1, inp1, key_padding_mask=padding_mask)[0].permute(1,0,2)
		inp = inp + self.attn_dropout(inp2)
		inp1 = self.ln2(inp)
		inp2 = self.l2(self.dropout1(F.relu(self.l1(inp1))))
		inp = inp + self.dropout2(inp2)
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

# ========================== #
#      Final Model - 1       #
# ========================== #	

class Final_ResGRU(nn.Module):
	def __init__(self, out_size, embed_size, hidden_size, max_seq_len=100, rnn_dropout=0.2, dnn_dropout=0.5, **kwargs):
		super(Final_ResGRU, self).__init__(**kwargs)
		self.out_size = out_size
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.max_seq_len = max_seq_len
		self.rnn_dropout = rnn_dropout
		self.dnn_dropout = dnn_dropout

		self.mlp_inp_size = sum(map(lambda x:4*x, hidden_size))

		for index, (e_size, h_size) in enumerate(zip(embed_size, hidden_size)):
			setattr(self, 'ResGRU_{}'.format(index), Extraction_ResGRU(e_size, h_size, max_seq_len=max_seq_len, dropout=rnn_dropout))

		self.bn = nn.BatchNorm1d(self.mlp_inp_size)
		self.dropout = nn.Dropout(p=dnn_dropout) 
		self.MLP = Output_MLP(self.mlp_inp_size, out_size, dropout=dnn_dropout)

	def forward(self, *args):
		batch_size, inp_len, buf = args[0].shape[0], args[-1], []
		for index, inp in enumerate(args[:-1]):
			inp = getattr(self, 'ResGRU_{}'.format(index))(inp, inp_len)
			out1 = inp[np.arange(len(inp_len)),inp_len-1,:]
			out2 = torch.stack([torch.max(inp[index,:l,:], dim=0)[0] for index, l in enumerate(inp_len)], dim=0)
			buf.append(torch.cat((out1, out2), dim=1))
		out = torch.cat(buf, dim=1)
		out = self.MLP(self.dropout(F.leaky_relu(self.bn(out), negative_slope=0.01)))
		return out

# ========================== #
#      Final Model - 2       #
# ========================== #

class Final_ResGRU_CNN(nn.Module):
	def __init__(self, out_size, embed_size, hidden_size, conv_channel=64, kernel_size=3, top_k=2, max_seq_len=100, rnn_dropout=0.2, dnn_dropout=0.5, **kwargs):
		super(Final_ResGRU_CNN, self).__init__(**kwargs)
		self.out_size = out_size
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.conv_channel = conv_channel
		self.kernel_size = kernel_size
		self.top_k = top_k
		self.max_seq_len = max_seq_len
		self.rnn_dropout = rnn_dropout
		self.dnn_dropout = dnn_dropout

		self.mlp_inp_size = sum(map(lambda x:4*x+conv_channel*top_k, hidden_size))

		for index, (e_size, h_size) in enumerate(zip(embed_size, hidden_size)):
			setattr(self, 'ResGRU_{}'.format(index), Extraction_ResGRU(e_size, h_size, max_seq_len=max_seq_len, dropout=rnn_dropout))
			setattr(self, 'CNN_{}'.format(index), Extraction_CNN(2*h_size, conv_channel, kernel_size))

		self.Kmax = Kmax_Pooling_Layer(2, top_k)
		self.bn = nn.BatchNorm1d(self.mlp_inp_size)
		self.dropout = nn.Dropout(p=dnn_dropout) 
		self.MLP = Output_MLP(self.mlp_inp_size, out_size, dropout=dnn_dropout)

	def forward(self, *args):
		batch_size, inp_len, buf = args[0].shape[0], args[-1], []
		for index, inp in enumerate(args[:-1]):
			inp = getattr(self, 'ResGRU_{}'.format(index))(inp, inp_len)
			out1 = inp[np.arange(len(inp_len)),inp_len-1,:]
			out2 = torch.stack([torch.max(inp[index,:l,:], dim=0)[0] for index, l in enumerate(inp_len)], dim=0)
			out3 = self.Kmax(getattr(self, 'CNN_{}'.format(index))(inp.permute(0,2,1))).view(batch_size, -1)
			buf.append(torch.cat((out1, out2, out3), dim=1))
		out = torch.cat(buf, dim=1)
		out = self.MLP(self.dropout(F.leaky_relu(self.bn(out), negative_slope=0.01)))
		return out

# ========================== #
#      Final Model - 3       #
# ========================== #

class Final_PreLN_Transformer(nn.Module):
	def __init__(self, out_size, embed_size, n_layer=1, n_head=4, intermediate_size=2048, device=None, max_seq_len=100, tf_dropout=0.1, rnn_dropout=0.2, dnn_dropout=0.5, **kwargs):
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

		self.mlp_inp_size = sum(map(lambda x:2*x, embed_size))

		for index, e_size in enumerate(embed_size):
			setattr(self, 'PreLN_TF_{}'.format(index), Extraction_PreLN_Transformer(n_layer, e_size, n_head, intermediate_size=intermediate_size, dropout=tf_dropout, device=self.device))
			setattr(self, 'LN_{}'.format(index), nn.LayerNorm(e_size))
			setattr(self, 'LSTM_{}'.format(index), nn.LSTM(input_size=e_size, hidden_size=e_size, batch_first=True))
			setattr(self, 'Dropout_{}'.format(index), nn.Dropout(p=rnn_dropout))

		self.bn = nn.BatchNorm1d(self.mlp_inp_size)
		self.dropout = nn.Dropout(p=dnn_dropout)
		self.MLP = Output_MLP(self.mlp_inp_size, out_size, dropout=dnn_dropout)

	def _pack(self, inp, inp_len):
		return nn.utils.rnn.pack_padded_sequence(inp, batch_first=True, lengths=inp_len, enforce_sorted=False)

	def _unpack(self, inp):
		return nn.utils.rnn.pad_packed_sequence(inp, batch_first=True, total_length=self.max_seq_len)[0]

	def forward(self, *args):
		batch_size, inp_len, buf = args[0].shape[0], args[-1], []
		for index, inp in enumerate(args[:-1]):
			inp = getattr(self, 'LN_{}'.format(index))(getattr(self, 'PreLN_TF_{}'.format(index))(inp, inp_len))
			inp = getattr(self, 'LSTM_{}'.format(index))(self._pack(inp, inp_len))[0]
			inp = getattr(self, 'Dropout_{}'.format(index))(self._unpack(inp))
			out1 = inp[np.arange(len(inp_len)),inp_len-1,:]
			out2 = torch.stack([torch.max(inp[index,:l,:], dim=0)[0] for index, l in enumerate(inp_len)], dim=0)
			buf.append(torch.cat((out1, out2), dim=1))
		out = torch.cat(buf, dim=1)
		out = self.MLP(self.dropout(F.leaky_relu(self.bn(out), negative_slope=0.01)))
		return out


