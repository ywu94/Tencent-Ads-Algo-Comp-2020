import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

class LSTM_Encoding_Layer(nn.Module):
	"""
	Input encoding layer for ESIM
	"""
	def __init__(self, embed_size, hidden_size, dropout=0.2, **kwargs):
		super(LSTM_Encoding_Layer, self).__init__(**kwargs)
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.dropout = dropout

		self.bi_lstm_layer = nn.LSTM(embed_size, hidden_size, bidirectional=True)
		self.dropout_layer = nn.Dropout(p=dropout)

	def forward(self, inp):
		out, _ = self.bi_lstm_layer(inp.permute(1,0,2))                                 # (seq_len, batch_size, 2*hidden_size)
		out = self.dropout_layer(out.permute(1,0,2))                                    # (batch_size, seq_len, 2*hidden_size)
		return out

class Local_Inference_Layer(nn.Module):
	"""
	Local inference layer for ESIM
	"""
	def __init__(self, device=None, **kwargs):
		super(Local_Inference_Layer, self).__init__(**kwargs)
		self.device = device if device else torch.device('cpu')

	def forward(self, inp_a, inp_b):
		align_weight = torch.einsum('ijk,imn->ijm', inp_a, inp_b)                       # (batch_size, seq_len_a, seq_len_b)
		align_weight_mask = (-1e9*torch.ones(align_weight.shape)).to(self.device)      
		align_weight = torch.where(align_weight==0, align_weight_mask, align_weight)                    
		attn_a = torch.einsum('iab,imn->ian', F.softmax(align_weight, dim=2), inp_b)    # (batch_size, seq_len_a, hidden_size)                                                       
		attn_b = torch.einsum('iab,ijk->ibk', F.softmax(align_weight, dim=1), inp_a)    # (batch_size, seq_len_b, hidden_size)
		inf_a = torch.cat((inp_a, attn_a, inp_a-attn_a, inp_a*attn_a), dim=2)           # (batch_size, seq_len_a, 4*hidden_size)
		inf_b = torch.cat((inp_b, attn_b, inp_b-attn_b, inp_b*attn_b), dim=2)           # (batch_size, seq_len_b, 4*hidden_size)                                    
		return inf_a, inf_b

class Inference_Composition_Layer(nn.Module):
	"""
	Inference composition layer of ESIM
	"""
	def __init__(self, input_size, hidden_size, max_seq_len=100, **kwargs):
		super(Inference_Composition_Layer, self).__init__(**kwargs)
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.max_seq_len = max_seq_len

		self.bi_lstm_layer = nn.LSTM(input_size, hidden_size, bidirectional=True)

	def forward(self, inp, inp_len):
		out, _ = self.bi_lstm_layer(inp.permute(1,0,2))                                 # (seq_len, batch_size, 2*hidden_size)
		out = out.permute(1,0,2)                                                        # (batch_size, seq_len, 2*hidden_size)
		max_buf, avg_buf = [], []
		for index, l in enumerate(inp_len):
			max_buf.append(torch.max(out[index,:l,:].permute(1,0), dim=1)[0])
			avg_buf.append(torch.mean(out[index,:l,:].permute(1,0), dim=1))
		max_out = torch.stack(max_buf, dim=0)                                            # (batch_size, 2*hidden_size)
		avg_out = torch.stack(avg_buf, dim=0)                                            # (batch_size, 2*hidden_size)
		return max_out, avg_out

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
		self.mlp_2 = nn.Linear(1024, 2048)
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

class ESIM_Classifier(nn.Module):
	"""
	Enhanced Sequential Inference Model for Classification
	"""
	def __init__(self, out_size, prem_embed_size, hypo_embed_size, hidden_size, device=None, max_seq_len=100, rnn_dropout=0.2, dnn_dropout=0.5, **kwargs):
		super(ESIM_Classifier, self).__init__(**kwargs)
		self.out_size = out_size
		self.prem_embed_size = prem_embed_size
		self.hypo_embed_size = hypo_embed_size
		self.hidden_size = hidden_size
		self.device = device if device else torch.device('cpu')
		self.max_seq_len = max_seq_len
		self.rnn_dropout = rnn_dropout
		self.dnn_dropout = dnn_dropout

		self.prem_encoder = LSTM_Encoding_Layer(prem_embed_size, hidden_size, dropout=rnn_dropout)
		self.hypo_encoder = LSTM_Encoding_Layer(hypo_embed_size, hidden_size, dropout=rnn_dropout)
		self.local_inference = Local_Inference_Layer(self.device)
		self.prem_inference = Inference_Composition_Layer(8*hidden_size, hidden_size, max_seq_len=max_seq_len)
		self.hypo_inference = Inference_Composition_Layer(8*hidden_size, hidden_size, max_seq_len=max_seq_len)
		self.inp_dropout = nn.Dropout(p=dnn_dropout)
		self.mlp_layer = MLP_Classification_Layer(8*hidden_size, out_size, dropout=dnn_dropout)


	def forward(self, inp_prem, inp_hypo, inp_len):
		inp_prem = self.prem_encoder(inp_prem.permute(1,0,2))                          # (seq_len_p, batch_size, 2*hidden_size)
		inp_prem = inp_prem.permute(1,0,2)                                             # (batch_size, seq_len_p, 2*hidden_size)
		inp_hypo = self.hypo_encoder(inp_hypo.permute(1,0,2))                          # (seq_len_h, batch_size, 2*hidden_size)
		inp_hypo = inp_hypo.permute(1,0,2)                                             # (batch_size, seq_len_h, 2*hidden_size)
		inf_prem, inf_hypo = self.local_inference(inp_prem, inp_hypo)                  # (batch_size, seq_len_p/h, 8*hidden_size)
		prem_max, prem_avg = self.prem_inference(inf_prem, inp_len)                    # (batch_size, 2*hidden_size)
		hypo_max, hypo_avg = self.hypo_inference(inf_hypo, inp_len)                    # (batch_size, 2*hidden_size)
		out = torch.cat((prem_max, prem_avg, hypo_max, hypo_avg), dim=1)               # (batch_size, 8*hidden_size)
		out = self.mlp_layer(self.inp_dropout(F.relu(out)))                            # (batch_size, out_size)
		return out                                            