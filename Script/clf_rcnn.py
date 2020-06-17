import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

class BiLSTM_Extraction_Layer(nn.Module):
	def __init__(self, embed_size, hidden_size, n_layer, max_seq_len=100, **kwargs):
		super(BiLSTM_Extraction_Layer, self).__init__(**kwargs)
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.n_layer = n_layer
		self.max_seq_len = max_seq_len
		
		self.bi_lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=n_layer, batch_first=True, bidirectional=True)
		
	def forward(self, inp, inp_len):
		inp = nn.utils.rnn.pack_padded_sequence(inp, batch_first=True, lengths=inp_len, enforce_sorted=False)
		inp = self.bi_lstm(inp)[0]
		inp = nn.utils.rnn.pad_packed_sequence(inp, batch_first=True, total_length=self.max_seq_len)[0]
		return inp

class CNN_Extraction_Layer(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
		super(CNN_Extraction_Layer, self).__init__(**kwargs)
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

class Kmax_Pooling_Layer(nn.Module):
	def __init__(self, dim, k, **kwargs):
		super(Kmax_Pooling_Layer, self).__init__(**kwargs)
		self.dim = dim
		self.k = k

	def forward(self, inp):
		index = inp.topk(self.k, dim=self.dim)[1].sort(dim=self.dim)[0]
		return inp.gather(self.dim, index)

class MLP_Output_Layer(nn.Module):
	def __init__(self, inp_size, out_size, dropout=0.5, ratio=4, **kwargs):
		super(MLP_Output_Layer, self).__init__(**kwargs)
		self.inp_size = inp_size
		self.out_size = out_size
		self.dropout = dropout
		
		self.mlp_1 = nn.Linear(inp_size, ratio*inp_size)
		self.batchnorm_1 = nn.BatchNorm1d(ratio*inp_size)
		self.mlp_dropout_1 = nn.Dropout(p=dropout)
		self.mlp_2 = nn.Linear(ratio*inp_size, inp_size)
		self.batchnorm_2 = nn.BatchNorm1d(inp_size)
		self.mlp_dropout_2 = nn.Dropout(p=dropout)
		self.mlp_3 = nn.Linear(inp_size, out_size)	

	def forward(self, inp):
		mlp_out = self.mlp_1(inp)                                                         # (batch_size, ratio*inp_size)
		mlp_out = self.mlp_dropout_1(F.relu(self.batchnorm_1(mlp_out)))                   # (batch_size, ratio*inp_size)
		mlp_out = self.mlp_2(mlp_out)                                                     # (batch_size, inp_size)
		mlp_out = self.mlp_dropout_2(F.relu(self.batchnorm_2(mlp_out)))                   # (batch_size, inp_size)
		mlp_out = self.mlp_3(mlp_out)                                                     # (batch_size, out_size)
		return mlp_out 

class RCNN_Classifier(nn.Module):
	def __init__(self, out_size, embed_size, hidden_size, n_layer=2, conv_channel=200, kernel_size=3, top_k=2, max_seq_len=100, dnn_dropout=0.5, **kwargs):
		super(RCNN_Classifier, self).__init__(**kwargs)
		self.out_size = out_size
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.n_layer = n_layer
		self.conv_channel = conv_channel
		self.kernel_size = kernel_size
		self.top_k = top_k
		self.max_seq_len = max_seq_len
		self.dnn_dropout = dnn_dropout

		self.lstm_extraction = BiLSTM_Extraction_Layer(embed_size, hidden_size, n_layer, max_seq_len=max_seq_len)
		self.cnn_extraction = CNN_Extraction_Layer(embed_size+hidden_size*2, conv_channel, kernel_size)
		self.kmax_pooling = Kmax_Pooling_Layer(2, top_k)
		self.mlp_output = MLP_Output_Layer(conv_channel*top_k, out_size, dropout=dnn_dropout)

	def forward(self, inp, inp_len):
		out = self.lstm_extraction(inp, inp_len)              # (batch_size, seq_len, 2*hidden_size)
		out = torch.cat((inp, out), dim=2).permute(0,2,1)     # (batch_size, 2*hidden_size+embed_size, seq_len)
		out = self.cnn_extraction(out)                        # (batch_size, conv_channel, seq_len+1-kernel_size)
		out = self.kmax_pooling(out)                          # (batch_size, conv_channel, top_k)
		reshaped = out.view(out.shape[0],-1)                  # (batch_size, conv_channel*top_k)
		out = self.mlp_output(reshaped)                       # (batch_size, out_size)
		return out

class Multi_Seq_RCNN_Classifier(nn.Module):
	def __init__(self, out_size, embed_size, hidden_size, n_layer=2, conv_channel=256, kernel_size=3, top_k=2, max_seq_len=100, dnn_dropout=0.5, **kwargs):
		super(Multi_Seq_RCNN_Classifier, self).__init__(**kwargs)
		self.out_size = out_size
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.n_layer = n_layer
		self.conv_channel = conv_channel
		self.kernel_size = kernel_size
		self.top_k = top_k
		self.max_seq_len = max_seq_len
		self.dnn_dropout = dnn_dropout

		for index, (e_size, h_size) in enumerate(zip(embed_size, hidden_size)):
			setattr(self, 'lstm_extraction_{}'.format(index), BiLSTM_Extraction_Layer(e_size, h_size, n_layer, max_seq_len=max_seq_len))
			setattr(self, 'cnn_extraction_{}'.format(index), CNN_Extraction_Layer(e_size+h_size*2, conv_channel, kernel_size))

		self.kmax_pooling = Kmax_Pooling_Layer(2, top_k)
		self.mlp_output = MLP_Output_Layer(len(embed_size)*conv_channel*top_k, out_size, dropout=dnn_dropout, ratio=2)

	def forward(self, *args):
		batch_size, inp_len, buf = args[0].shape[0], args[-1], []
		for index, inp in enumerate(args[:-1]):
			out = getattr(self, 'lstm_extraction_{}'.format(index))(inp, inp_len)
			out = torch.cat((inp, out), dim=2).permute(0,2,1)
			reshaped = self.kmax_pooling(getattr(self, 'cnn_extraction_{}'.format(index))(out)).view(batch_size, -1)
			buf.append(reshaped)
		out = torch.cat(buf, dim=1)
		out = self.mlp_output(out)
		return out




