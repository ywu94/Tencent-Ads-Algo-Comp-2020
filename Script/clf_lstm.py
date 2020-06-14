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
			setattr(self, 'extraction_layer_{}'.format(index), LSTM_Extraction_Layer(e_size, h_size, rnn_dropout=rnn_dropout, mlp_dropout=mlp_dropout))
		self.classification_layer = MLP_Classification_Layer(self.mlp_inp_size, out_size, dropout=mlp_dropout)
		
	def forward(self, *args):
		assert len(args)==self.n_extraction+1
		
		extract_buffer = [getattr(self, 'extraction_layer_{}'.format(index))(inp_embed, args[-1]) for index, inp_embed in enumerate(args[:-1])]
		out = torch.cat(extract_buffer, 1)
		out = self.classification_layer(out)
		
		return out

class LSTM_Extraction_V2_Layer(nn.Module):
	"""
	LSTM feature extration layer
	- Layer 1: BiLSTM + Dropout + Layernorm
	- Layer 2: LSTM with Residual Connection + Dropout + Layernorm
	- Layer 3: LSTM + Batchnorm + ReLU + Dropout
	- V2: Add max pooling for feature extraction
	"""
	def __init__(self, embed_size, lstm_hidden_size, max_seq_len=100, rnn_dropout=0.2, mlp_dropout=0.4, **kwargs):
		super(LSTM_Extraction_V2_Layer, self).__init__(**kwargs)
		self.embed_size = embed_size
		self.lstm_hidden_size = lstm_hidden_size
		self.max_seq_len = max_seq_len
		self.rnn_dropout = rnn_dropout
		self.mlp_dropout = mlp_dropout
		
		self.bi_lstm = nn.LSTM(input_size=embed_size, hidden_size=lstm_hidden_size, bias=True, bidirectional=True)
		self.rnn_dropout_1 = nn.Dropout(p=rnn_dropout)
		self.layernorm_1 = nn.LayerNorm(2*lstm_hidden_size)
		self.lstm_1 = nn.LSTM(input_size=2*lstm_hidden_size, hidden_size=2*lstm_hidden_size)
		self.rnn_dropout_2 = nn.Dropout(p=rnn_dropout)
		self.layernorm_2 = nn.LayerNorm(2*lstm_hidden_size)
		self.lstm_2 = nn.LSTM(input_size=2*lstm_hidden_size, hidden_size=2*lstm_hidden_size)
		self.batchnorm = nn.BatchNorm1d(4*lstm_hidden_size)
		self.mlp_dropout = nn.Dropout(p=mlp_dropout)
		self.max_pooling = nn.MaxPool1d(max_seq_len)
		
	def forward(self, inp_embed, inp_last_idx):
		bilstm_out, _ = self.bi_lstm(inp_embed.permute(1,0,2))                            # (max_seq_length, batch_size, embed_size) -> (max_seq_length, batch_size, 2*lstm_hidden_size)
		bilstm_out = self.layernorm_1(self.rnn_dropout_1(bilstm_out))                     # (max_seq_length, batch_size, 2*lstm_hidden_size)
		lstm_out, _ = self.lstm_1(bilstm_out)                                             # (max_seq_length, batch_size, 2*lstm_hidden_size)
		lstm_out = self.rnn_dropout_2(lstm_out)                                           # (max_seq_length, batch_size, 2*lstm_hidden_size)
		lstm_out = self.layernorm_2(lstm_out+bilstm_out)                                  # (max_seq_length, batch_size, 2*lstm_hidden_size)
		lstm_out, _ = self.lstm_2(lstm_out)                                               # (max_seq_length, batch_size, 2*lstm_hidden_size)
		out_1 = lstm_out.permute(1,0,2)[np.arange(len(inp_last_idx)), inp_last_idx,:]     # (batch_size, 2*lstm_hidden_size)
		out_2 = self.max_pooling(lstm_out.permute(1,2,0)).squeeze(2)                      # (batch_size, 2*lstm_hidden_size)
		out = self.mlp_dropout(F.relu(self.batchnorm(torch.cat((out_1, out_2), dim=1))))  # (batch_size, 4*lstm_hidden_size)
		return out

class MLP_Classification_V2_Layer(nn.Module):
	"""
	Multilayer Perception Classification Layer
	- Layer 1: Linear + Batchnorm + ReLU + Dropout
	- Layer 2: Linear + Batchnorm + ReLU + Dropout
	- Layer 3: Linear
	"""
	def __init__(self, inp_size, out_size, dropout=0.5, **kwargs):
		super(MLP_Classification_V2_Layer, self).__init__(**kwargs)
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

class Multi_Seq_LSTM_V2_Classifier(nn.Module):
	"""
	Use separate LSTM extractor to handle different sequences, concat them and feed backto multilayer perception classifier.
	- V2: Add max pooling for feature extraction
	"""
	def __init__(self, embed_size, lstm_hidden_size, out_size, max_seq_len=100, rnn_dropout=0.2, mlp_dropout=0.5, **kwargs):
		super(Multi_Seq_LSTM_V2_Classifier, self).__init__(**kwargs)
		assert isinstance(embed_size, list) and isinstance(lstm_hidden_size, list) and len(embed_size)==len(lstm_hidden_size)
		
		self.embed_size = embed_size
		self.lstm_hidden_size = lstm_hidden_size
		self.out_size = out_size
		self.max_seq_len = max_seq_len
		self.rnn_dropout = rnn_dropout
		self.mlp_dropout = mlp_dropout
		
		self.n_extraction = len(embed_size)
		self.mlp_inp_size = sum(map(lambda x:4*x, lstm_hidden_size))
		
		for index, (e_size, h_size) in enumerate(zip(embed_size, lstm_hidden_size)):
			setattr(self, 'extraction_layer_{}'.format(index), LSTM_Extraction_V2_Layer(e_size, h_size, max_seq_len=max_seq_len, rnn_dropout=rnn_dropout, mlp_dropout=mlp_dropout))
		self.classification_layer = MLP_Classification_V2_Layer(self.mlp_inp_size, out_size, dropout=mlp_dropout)
		
	def forward(self, *args):
		assert len(args)==self.n_extraction+1
		
		extract_buffer = [getattr(self, 'extraction_layer_{}'.format(index))(inp_embed, args[-1]) for index, inp_embed in enumerate(args[:-1])]
		out = torch.cat(extract_buffer, 1)
		out = self.classification_layer(out)
		
		return out

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

class LSTM_Attn_Encoder_Layer(nn.Module):
	"""
	Google Neural Machine Translation - Encoder (Shallow Version)
	"""
	def __init__(self, input_size, hidden_size, dropout=0.1, **kwargs):
		super(LSTM_Attn_Encoder_Layer, self).__init__(**kwargs)

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.dropout = dropout

		self.l1_bilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, bias=True, bidirectional=True)
		self.l1_dropout = nn.Dropout(p=dropout)
		self.l2_lstm = nn.LSTM(input_size=hidden_size*2, hidden_size=hidden_size, bias=True)
		self.l2_dropout = nn.Dropout(p=dropout)

	def forward(self, inp, inp_len):
		batch_size, total_length, _ = inp.shape
		inp = nn.utils.rnn.pack_padded_sequence(inp, batch_first=True, lengths=inp_len, enforce_sorted=False)
		out, (h, c) = self.l1_bilstm(inp)
		backward_hidden_state = h.view(1, 2, batch_size, self.hidden_size)[:,1,:,:].squeeze(0)                                                      # (num_direction, batch_size, enc_hidden_size)
		backward_cell_state = c.view(1, 2, batch_size, self.hidden_size)[:,1,:,:].squeeze(0)                                                        # (num_direction, batch_size, enc_hidden_size)
		out = self.l1_dropout(nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=total_length)[0])
		out = nn.utils.rnn.pack_padded_sequence(out, batch_first=True, lengths=inp_len, enforce_sorted=False)
		out, _ = self.l2_lstm(out)
		out = self.l2_dropout(nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=total_length)[0])
		return out, backward_hidden_state, backward_cell_state

class LSTM_Attn_Decoder_Layer(nn.Module):
	"""
	Google Neural Machine Translation - Decoder (Shallow Version)
	"""
	def __init__(self, hidden_size, dropout=0.1, device=None, **kwargs):
		super(LSTM_Attn_Decoder_Layer, self).__init__(**kwargs)
		self.hidden_size = hidden_size
		self.dropout = dropout
		self.device = device if device else torch.device('cpu')

		self.attention_calc = Additive_Attention_Layer(hidden_size)
		self.l1_lstm_cell = nn.LSTMCell(input_size=2*hidden_size, hidden_size=hidden_size, bias=True)
		self.l1_dropout = nn.Dropout(p=dropout)
		self.l2_lstm = nn.LSTM(input_size=2*hidden_size, hidden_size=hidden_size, batch_first=True, bias=True)
		self.l2_dropout = nn.Dropout(p=dropout)

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
			inp = torch.cat((inp, context_vector), dim=1)                                                                                             # (batch_size, 2*hidden_size)
			decoder_h, decoder_c = self.l1_lstm_cell(inp, (decoder_c, decoder_h))
			decoder_hidden_states_buf.append(decoder_h)
		decoder_context_vectors = torch.stack(decoder_context_vectors_buf, dim=1)                                                                     # (batch_size, seq_len, hidden_size)
		decoder_hidden_states = torch.stack(decoder_hidden_states_buf, dim=1)                                                                         # (batch_size, seq_len, hidden_size)
		decoder_hidden_states = self.l1_dropout(torch.cat((decoder_hidden_states, decoder_context_vectors), dim=2))                                   # (batch_size, seq_len, 2*hidden_size)
		decoder_hidden_states = nn.utils.rnn.pack_padded_sequence(decoder_hidden_states, batch_first=True, lengths=inp_len, enforce_sorted=False)
		decoder_hidden_states, _ = self.l2_lstm(decoder_hidden_states)
		decoder_hidden_states = nn.utils.rnn.pad_packed_sequence(decoder_hidden_states, batch_first=True, total_length=seq_len)[0]
		decoder_hidden_states = self.l2_dropout(decoder_hidden_states)
		return decoder_hidden_states

class LSTM_Attn_Extraction_Layer(nn.Module):
	"""
	Seq2Seq feature extration layer based on Google Neural Machine Translation (Shallow Version)
	"""
	def __init__(self, embed_size, hidden_size, device=None, dropout=0.1, **kwargs):
		super(LSTM_Attn_Extraction_Layer, self).__init__(**kwargs)
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.device = device if device else torch.device('cpu')
		self.dropout = dropout

		self.encoder = LSTM_Attn_Encoder_Layer(embed_size, hidden_size)
		self.decoder = LSTM_Attn_Decoder_Layer(hidden_size, device=self.device)

	def forward(self, inp, inp_len):
		encoder_hidden_states, backward_hidden_state, backward_cell_state = self.encoder(inp, inp_len)
		decoder_hidden_states = self.decoder(encoder_hidden_states, backward_hidden_state, backward_cell_state, inp_len)
		return decoder_hidden_states

class LSTM_Attn_Conv_Classifier(nn.Module):
	"""
	Use LSTM extractor to handle sequence, use conv + pool to extract information, then use mlp classifier.
	"""
	def __init__(self, embed_size, hidden_size, out_size, seq_len=100, device=None, rnn_dropout=0.2, dnn_dropout=0.5, **kwargs):
		super(LSTM_Attn_Conv_Classifier, self).__init__(**kwargs)
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.seq_len = seq_len
		self.device = device if device else torch.device('cpu')
		self.rnn_dropout = rnn_dropout
		self.dnn_dropout = dnn_dropout

		self.extraction_layer = LSTM_Attn_Extraction_Layer(embed_size, hidden_size, device, rnn_dropout)
		self.conv_layer = nn.Conv1d(seq_len, 128, 1)
		self.inp_dropout = nn.Dropout(p=dnn_dropout)
		self.mlp_layer = MLP_Classification_Layer(2*hidden_size, out_size, dropout=dnn_dropout)

	def forward(self, inp, inp_len):
		inp = self.extraction_layer(inp, inp_len)                            # (batch_size, seq_len, hidden_size)
		pol = torch.max(inp.permute(0, 2, 1), dim=2)[0]                      # (batch_size, hidden_size)
		con = torch.mean(self.conv_layer(inp).permute(0, 2, 1), dim=2)       # (batch_size, hidden_size)
		inp = torch.cat((pol, con), dim=1)                                   # (batch_size, hidden_size*2)
		out = self.mlp_layer(self.inp_dropout(F.relu(inp)))                  # (batch_size, out_size)
		return out

class Multi_Seq_LSTM_Attn_Conv_Classifier(nn.Module):
	"""
	Use multiple LSTM extractors to handle different sequences, use conv + pool to extract information, then use mlp classifier.
	"""
	def __init__(self, embed_size, hidden_size, out_size, seq_len=100, device=None, rnn_dropout=0.2, dnn_dropout=0.5, **kwargs):
		super(Multi_Seq_LSTM_Attn_Conv_Classifier, self).__init__(**kwargs)
		assert isinstance(embed_size, list) and isinstance(hidden_size, list) and len(embed_size)==len(hidden_size)

		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.out_size = out_size
		self.seq_len = seq_len
		self.device = device if device else torch.device('cpu')
		self.rnn_dropout = rnn_dropout
		self.dnn_dropout = dnn_dropout

		self.n_extraction = len(embed_size)
		self.mlp_inp_size = sum(map(lambda x:2*x, hidden_size))

		for index, (e_size, h_size) in enumerate(zip(embed_size, hidden_size)):
			setattr(self, 'extraction_layer_{}'.format(index), LSTM_Attn_Extraction_Layer(e_size, h_size, device, rnn_dropout))
			setattr(self, 'conv_layer_{}'.format(index), nn.Conv1d(seq_len, 128, 1))

		self.inp_dropout = nn.Dropout(p=dnn_dropout)

		for index, o_size in enumerate(out_size):
			setattr(self, 'mlp_layer_{}'.format(index), MLP_Classification_Layer(self.mlp_inp_size, o_size, dropout=dnn_dropout))

	def forward(self, *args):
		assert len(args)==self.n_extraction+1
		ext_buf = [getattr(self, 'extraction_layer_{}'.format(index))(inp_embed, args[-1]) for index, inp_embed in enumerate(args[:-1])]
		pol_buf = [torch.max(i.permute(0, 2, 1), dim=2)[0] for i in ext_buf]
		con_buf = [torch.mean(getattr(self, 'conv_layer_{}'.format(index))(i).permute(0,2,1), dim=2) for index, i in enumerate(ext_buf)]
		inp = torch.cat(pol_buf+con_buf, dim=1)
		out = [getattr(self, 'mlp_layer_{}'.format(index))(inp) for index in range(len(self.out_size))]
		return out






