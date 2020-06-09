"""
Reference:
[1] https://gist.github.com/shreydesai/fc20a99b56392930b34489e20a0c7f88
"""

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

class Additive_Attention(nn.Module):
	"""
	Additive attention used in GNMT
	"""
	def __init__(self, hidden_size, **kwargs):
		super(Additive_Attention, self).__init__(**kwargs)
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

class Res_LSTM_Layer(nn.Module):
	"""
	Multi-layer unidirectional with residual connection.
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
		for index in range(self.n_layer):
			out = nn.utils.rnn.pack_padded_sequence(inp, batch_first=True, lengths=inp_len, enforce_sorted=False)
			out, _ = getattr(self, 'lstm_{}'.format(index))(out)
			inp = getattr(self, 'dropout_{}'.format(index))(torch.add(nn.utils.rnn.pad_packed_sequence(out, batch_first=True)[0], inp))
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
		self.l2_lstm = nn.LSTM(input_size=input_size*2, hidden_size=hidden_size, bias=True)
		self.l2_dropout = nn.Dropout(p=dropout)
		self.res_lstm = Res_LSTM_Layer(n_layer-2, hidden_size, dropout=dropout)

	def forward(self, inp, inp_len):
		batch_size = inp.shape[0]
		inp = nn.utils.rnn.pack_padded_sequence(inp, batch_first=True, lengths=inp_len, enforce_sorted=False)
		out, (h, c) = self.l1_bilstm(inp)
		backward_hidden_state = h.view(1, 2, batch_size, self.hidden_size)[:,1,:,:]                                                # (num_direction, batch_size, enc_hidden_size)
		backward_cell_state = c.view(1, 2, batch_size, self.hidden_size)[:,1,:,:]                                                  # (num_direction, batch_size, enc_hidden_size)
		out = self.l1_dropout(nn.utils.rnn.pad_packed_sequence(out, batch_first=True)[0])
		out = nn.utils.rnn.pack_padded_sequence(out, batch_first=True, lengths=inp_len, enforce_sorted=False)
		out, _ = self.l2_lstm(out)
		out = self.l2_dropout(nn.utils.rnn.pad_packed_sequence(out, batch_first=True)[0])
		out = self.res_lstm(out, inp_len)
		return out, backward_hidden_state, backward_cell_state

class GNMT_Decoder_Layer(nn.Module):
	"""
	Google Neural Machine Translation - Decoder
	"""
	def __init__(self, n_layer, hidden_size, dropout=0.1, **kwargs):
		super(GNMT_Decoder_Layer, self).__init__(**kwargs)
		self.n_layer = n_layer
		self.hidden_size = hidden_size
		self.dropout = dropout

	def forward(self, enc_hidden_state, backward_hidden_state, backward_cell_state, inp_len):

		

                                                      


class GNMT_Extraction_Layer(nn.Module):
	"""
	Feature extration layer based on Google Neural Machine Translation.
	"""
	def __init__(self, embed_size, enc_n_layer, enc_hidden_size, dec_n_layer, dec_hidden_size, rnn_dropout=0.1, dnn_dropout=0.4, **kwargs):
		super(LSTM_Attention_Extraction_Layer, self).__init__(**kwargs)
		assert enc_n_layer>=3 
		self.embed_size = embed_size
		self.enc_n_layer = enc_n_layer
		self.enc_hidden_size = enc_hidden_size
		self.dec_hidden_size = dec_hidden_size
		self.rnn_dropout = rnn_dropout
		self.dnn_dropout = dnn_dropout

		self.enc_bilstm_1 = nn.LSTM(input_size=embed_size, hidden_size=enc_hidden_size, batch_first=True, bias=True, bidirectional=True)
		self.rnn_dropout_1 = nn.Dropout(p=rnn_dropout)
		self.enc_lstm_2 = nn.LSTM(input_size=enc_hidden_size*2, hidden_size=enc_hidden_size, bias=True)
		self.rnn_dropout_2 = nn.Dropout(p=rnn_dropout)
		self.enc_lstm_3 = nn.LSTM(input_size=enc_hidden_size, hidden_size=enc_hidden_size, bias=True)
		self.rnn_dropout_3 = nn.Dropout(p=rnn_dropout)
		self.enc_lstm_4 = nn.LSTM(input_size=enc_hidden_size, hidden_size=enc_hidden_size, bias=True)
		self.rnn_dropout_4 = nn.Dropout(p=rnn_dropout)


	def forward(self, inp_embed, inp_last_idx):
		batch_size = inp_embed.shape[0]
		inp_embed = torch.nn.utils.rnn.pack_padded_sequence(inp_embed, batch_first=True, lengths=inp_last_idx+1, enforce_sorted=False)
		out, (h, c) = self.enc_bilstm_1(inp_embed)
		backward_hidden_state = h.view(1, 2, batch_size, self.enc_hidden_size)[:,1,:,:]                                                # (num_direction, batch_size, enc_hidden_size)
		backward_cell_state = c.view(1, 2, batch_size, self.enc_hidden_size)[:,1,:,:]                                                  # (num_direction, batch_size, enc_hidden_size)
		out_unpacked = self.rnn_dropout_1(nn.utils.rnn.pad_packed_sequence(out, batch_first=True)[0])                                  # (batch_size, seq_len, 2*enc_hidden_size)
		out = torch.nn.utils.rnn.pack_padded_sequence(out_unpacked, batch_first=True, lengths=inp_last_idx+1, enforce_sorted=False)
		out, _ = self.enc_lstm_2(out)
		out_unpacked = self.rnn_dropout_2(nn.utils.rnn.pad_packed_sequence(out, batch_first=True)[0])                                  # (batch_size, seq_len, enc_hidden_size)
		out = torch.nn.utils.rnn.pack_padded_sequence(out_unpacked, batch_first=True, lengths=inp_last_idx+1, enforce_sorted=False)
		out, _ = self.enc_lstm_3(out)
		out_unpacked = self.rnn_dropout_3(torch.add(nn.utils.rnn.pad_packed_sequence(out, batch_first=True)[0], out_unpacked))         # (batch_size, seq_len, enc_hidden_size)
		return out_unpacked                                                             




		# (seq_len, batch_size, enc_hidden_size)



		inp_embed, _ = torch.add(self.enc_lstm_3(self.rnn_dropout_2(inp_embed)), inp_embed)                    # (seq_len, batch_size, enc_hidden_size)
		inp_embed, _ = torch.add(self.enc)






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


class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

	def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights


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