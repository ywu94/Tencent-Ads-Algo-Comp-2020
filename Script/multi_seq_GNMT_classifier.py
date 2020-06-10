import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

class Res_LSTM_Layer(nn.Module):
	"""
	Multi-layer unidirectional LSTM with residual connection.
	"""
	def __init__(self, n_layer, hidden_size, dropout=0.1, **kwargs):
		super(Res_LSTM_Layer, self).__init__(**kwargs)
		self.n_layer = n_layer
		self.hidden_size = hidden_size
		self.dropout = dropout

		for index in range(n_layer):
			setattr(self, 'lstm_{}'.format(index), nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True, bias=True))
			setattr(self, 'dropout_{}'.format(index), nn.Dropout(p=dropout))

	def forward(self, inp, inp_len):
		_, total_length, _ = inp.shape
		for index in range(self.n_layer):
			out = nn.utils.rnn.pack_padded_sequence(inp, batch_first=True, lengths=inp_len, enforce_sorted=False)
			out, _ = getattr(self, 'lstm_{}'.format(index))(out)
			out = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=total_length)[0]
			inp = getattr(self, 'dropout_{}'.format(index))(torch.add(out, inp))
		return inp

class GNMT_Encoder_Layer(nn.Module):
	"""
	Google Neural Machine Translation - Encoder
	"""
	def __init__(self, input_size, n_layer, hidden_size, dropout=0.1, **kwargs):
		super(GNMT_Encoder_Layer, self).__init__(**kwargs)
		assert n_layer >= 3

		self.input_size = input_size
		self.n_layer = n_layer
		self.hidden_size = hidden_size
		self.dropout = dropout

		self.l1_bilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, bias=True, bidirectional=True)
		self.l1_dropout = nn.Dropout(p=dropout)
		self.l2_lstm = nn.LSTM(input_size=hidden_size*2, hidden_size=hidden_size, bias=True)
		self.l2_dropout = nn.Dropout(p=dropout)
		self.res_lstm = Res_LSTM_Layer(n_layer-2, hidden_size, dropout=dropout)

	def forward(self, inp, inp_len):
		batch_size, total_length, _ = inp.shape
		inp = nn.utils.rnn.pack_padded_sequence(inp, batch_first=True, lengths=inp_len, enforce_sorted=False)
		out, (h, c) = self.l1_bilstm(inp)
		backward_hidden_state = h.view(1, 2, batch_size, self.hidden_size)[:,1,:,:].squeeze(0)              # (num_direction, batch_size, enc_hidden_size)
		backward_cell_state = c.view(1, 2, batch_size, self.hidden_size)[:,1,:,:].squeeze(0)                # (num_direction, batch_size, enc_hidden_size)
		out = self.l1_dropout(nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=total_length)[0])
		out = nn.utils.rnn.pack_padded_sequence(out, batch_first=True, lengths=inp_len, enforce_sorted=False)
		out, _ = self.l2_lstm(out)
		out = self.l2_dropout(nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=total_length)[0])
		out = self.res_lstm(out, inp_len)
		return out, backward_hidden_state, backward_cell_state

class Additive_Attention_Layer(nn.Module):
	"""
	Additive attention used in GNMT
	"""
	def __init__(self, hidden_size, **kwargs):
		super(Additive_Attention_Layer, self).__init__(**kwargs)
		self.hidden_size = hidden_size

		self.W = nn.Linear(hidden_size*2, hidden_size)
		self.tanh = nn.Tanh()
		self.V = nn.Parameter(torch.Tensor(1, hidden_size))
		self.softmax = nn.Softmax(dim=2)

		nn.init.normal_(self.V, 0, 0.1)

	def forward(self, query, values, mask):
		"""
		: query:  (batch_size, hidden_size)
		: values: (batch_size, seq_len, hidden_size)
		: mask:   (batch_size, seq_len)
		"""
		batch_size, seq_len, hidden_size = values.shape

		query = query.unsqueeze(1).expand(-1, seq_len, -1)
		score = self.tanh(self.W(torch.cat((query, values), dim=2)))                              # (batch_size, seq_len, hidden_size)
		score = torch.bmm(self.V.squeeze(1).expand(batch_size, -1, -1), score.permute(0,2,1))     # (batch_size, 1, seq_len)
		score = self.softmax(torch.add(score, mask.unsqueeze(1)))                                 # (batch_size, 1, seq_len)
		context = torch.bmm(score, values).squeeze(1)                                             # (batch_size, hidden_size)

		return context

class Res_Attn_LSTM_Layer(nn.Module):
	"""
	Multi-layer unidirectional LSTM with residual connection and attention.
	"""
	def __init__(self, n_layer, hidden_size, dropout=0.1, **kwargs):
		super(Res_Attn_LSTM_Layer, self).__init__(**kwargs)
		self.n_layer = n_layer
		self.hidden_size = hidden_size
		self.dropout = dropout

		for index in range(n_layer):
			setattr(self, 'lstm_{}'.format(index), nn.LSTM(input_size=2*hidden_size, hidden_size=hidden_size, batch_first=True, bias=True))
			setattr(self, 'dropout_{}'.format(index), nn.Dropout(p=dropout))

	def forward(self, hidden_states, context_vectors, inp_len):
		_, total_length, _ = hidden_states.shape
		for index in range(self.n_layer):
			out = nn.utils.rnn.pack_padded_sequence(torch.cat((hidden_states, context_vectors), dim=2), batch_first=True, lengths=inp_len, enforce_sorted=False)
			out, _ = getattr(self, 'lstm_{}'.format(index))(out)
			out = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=total_length)[0]
			hidden_states = getattr(self, 'dropout_{}'.format(index))(torch.add(out, hidden_states))
		return hidden_states

class GNMT_Decoder_Layer(nn.Module):
	"""
	Google Neural Machine Translation - Decoder
	"""
	def __init__(self, n_layer, hidden_size, dropout=0.1, device=None, **kwargs):
		super(GNMT_Decoder_Layer, self).__init__(**kwargs)
		assert n_layer>=3

		self.n_layer = n_layer
		self.hidden_size = hidden_size
		self.dropout = dropout
		self.device = device if device else torch.device('cpu')

		self.attention_calc = Additive_Attention_Layer(hidden_size)
		self.l1_lstm_cell = nn.LSTMCell(input_size=2*hidden_size, hidden_size=hidden_size, bias=True)
		self.l1_dropout = nn.Dropout(p=dropout)
		self.l2_lstm = nn.LSTM(input_size=2*hidden_size, hidden_size=hidden_size, batch_first=True, bias=True)
		self.l2_dropout = nn.Dropout(p=dropout)
		self.res_attn_lstm = Res_Attn_LSTM_Layer(n_layer-2, hidden_size, dropout=dropout)

	def get_attention_mask(self, inp_len, batch_size, seq_len):
		mask = np.ones((batch_size, seq_len))
		for index, l in enumerate(inp_len):
			mask[index,:l] = 0
		mask *= -1e9
		return torch.from_numpy(mask).float().to(self.device)

	def forward(self, enc_hidden_states, backward_hidden_state, backward_cell_state, inp_len):
		batch_size, seq_len, _ = enc_hidden_states.shape
		attention_mask = self.get_attention_mask(inp_len, batch_size, seq_len)
		enc_hidden_states = enc_hidden_states.permute(1,0,2)                                                                                          # (seq_len, batch_size, hidden_size)
		decoder_hidden_states_buf =  []
		decoder_context_vectors_buf = []
		decoder_h, decoder_c = backward_hidden_state, backward_cell_state
		for step in range(seq_len):
			inp = enc_hidden_states[step]                        
			context_vector = self.attention_calc(inp, enc_hidden_states.permute(1,0,2), attention_mask)                                               # (batch_size, hidden_size)
			decoder_context_vectors_buf.append(context_vector)
			inp = torch.cat((inp, context_vector), dim=1)                                                                                                    # (batch_size, 2*hidden_size)
			decoder_h, decoder_c = self.l1_lstm_cell(inp, (decoder_c, decoder_h))
			decoder_hidden_states_buf.append(decoder_h)
		decoder_context_vectors = torch.stack(decoder_context_vectors_buf, dim=1)                                                                     # (batch_size, seq_len, hidden_size)
		decoder_hidden_states = torch.stack(decoder_hidden_states_buf, dim=1)                                                                         # (batch_size, seq_len, hidden_size)
		decoder_hidden_states = self.l1_dropout(torch.cat((decoder_hidden_states, decoder_context_vectors), dim=2))                                   # (batch_size, seq_len, 2*hidden_size)
		decoder_hidden_states = nn.utils.rnn.pack_padded_sequence(decoder_hidden_states, batch_first=True, lengths=inp_len, enforce_sorted=False)
		decoder_hidden_states, _ = self.l2_lstm(decoder_hidden_states)
		decoder_hidden_states = nn.utils.rnn.pad_packed_sequence(decoder_hidden_states, batch_first=True, total_length=seq_len)[0]
		decoder_hidden_states = self.l2_dropout(decoder_hidden_states)
		decoder_hidden_states = self.res_attn_lstm(decoder_hidden_states, decoder_context_vectors, inp_len)                                                    # (batch_size, seq_len, hidden_size)
		return decoder_hidden_states

class GNMT_Extraction_Layer(nn.Module):
	"""
	Seq2Seq feature extration layer based on Google Neural Machine Translation.
	"""
	def __init__(self, embed_size, hidden_size, n_enc_layer, n_dec_layer, device=None, dropout=0.1, **kwargs):
		super(GNMT_Extraction_Layer, self).__init__(**kwargs)
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.n_enc_layer = n_enc_layer
		self.n_dec_layer = n_dec_layer
		self.device = device if device else torch.device('cpu')
		self.dropout = dropout

		self.encoder = GNMT_Encoder_Layer(embed_size, n_enc_layer, hidden_size)
		self.decoder = GNMT_Decoder_Layer(n_dec_layer, hidden_size, device=self.device)

	def forward(self, inp, inp_len):
		encoder_hidden_states, backward_hidden_state, backward_cell_state = self.encoder(inp, inp_len)
		decoder_hidden_states = self.decoder(encoder_hidden_states, backward_hidden_state, backward_cell_state, inp_len)
		return decoder_hidden_states

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
		
		self.mlp_1 = nn.Linear(inp_size, 2048)
		self.batchnorm_1 = nn.BatchNorm1d(2048)
		self.mlp_dropout_1 = nn.Dropout(p=dropout)
		self.mlp_2 = nn.Linear(2048, 1024)
		self.batchnorm_2 = nn.BatchNorm1d(1024)
		self.mlp_dropout_2 = nn.Dropout(p=dropout)
		self.mlp_3 = nn.Linear(1024, out_size)
		
	def forward(self, inp):
		mlp_out = self.mlp_1(inp)                                                         # (batch_size, 1024)
		mlp_out = self.mlp_dropout_1(F.relu(self.batchnorm_1(mlp_out)))                   # (batch_size, 1024)
		mlp_out = self.mlp_2(mlp_out)                                                     # (batch_size, 512)
		mlp_out = self.mlp_dropout_2(F.relu(self.batchnorm_2(mlp_out)))                   # (batch_size, 512)
		mlp_out = self.mlp_3(mlp_out)                                                     # (batch_size, out_size)
		return mlp_out   

class Multi_Seq_GNMT_Classifier(nn.Module):
	"""
	Use GNMT for Seq2Seq feature extraction, apply max pooling & pick last state, then use multilayer perception for classification
	- Multi sequence input version
	"""
	def __init__(self, out_size, embed_size, hidden_size, n_enc_layer, n_dec_layer, device=None, rnn_dropout=0.1, dnn_dropout=0.4, **kwargs):
		super(Multi_Seq_GNMT_Classifier, self).__init__()
		assert isinstance(embed_size, list) and isinstance(hidden_size, list) and len(embed_size)==len(hidden_size)
		self.out_size = out_size
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.n_enc_layer = n_enc_layer
		self.n_dec_layer = n_dec_layer
		self.device = device if device else torch.device('cpu')
		self.rnn_dropout = rnn_dropout
		self.dnn_dropout = dnn_dropout

		self.n_extraction = len(embed_size)
		self.mlp_inp_size = sum(map(lambda x:2*x, hidden_size))

		for index, (e_size, h_size) in enumerate(zip(embed_size, hidden_size)):
			setattr(self, 'GNMT_layer_{}'.format(index), GNMT_Extraction_Layer(e_size, h_size, n_enc_layer, n_dec_layer, device=device, dropout=rnn_dropout))
		self.mlp_layer = MLP_Classification_Layer(self.mlp_inp_size, out_size, dropout=dnn_dropout)

	def forward(self, *args):
		assert len(args)==self.n_extraction+1
		out_buf, inp_len = [], args[-1]
		for index, inp in enumerate(args[:-1]):
			inp = getattr(self, 'GNMT_layer_{}'.format(index))(inp, inp_len)  
			max_pool_buf, last_buf = [], []
			for batch_idx, l in enumerate(inp_len):
				max_pool_buf.append(torch.max(inp[batch_idx,:l], dim=0)[0])
				last_buf.append(inp[batch_idx, l-1])
			out_buf.append(torch.cat((torch.stack(max_pool_buf, dim=0), torch.stack(last_buf, dim=0)), dim=1)) 
		out = torch.cat(out_buf, dim=1)
		out = self.mlp_layer(out)                                                     
		return out