"""
    Model file

    author: Zeyu Li <zeyuli@ucla.ed> or <zyli@cs.ucla.edu>

    Implemented model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from collections import OrderedDict

class Embed(nn.Module):

    def __init__(self
                 , vocab_size
                 , embedding_dim
                 , lstm_layers):
        super(Embed, self).__init__()
        self.emb_dim = embedding_dim
        print("vocab_size", vocab_size)
        self.lstm_layers = lstm_layers
        self.ru_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=False)
        self.rv_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=False)
        self.au_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=False)
        self.av_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=False)

        self.init_emb()
        self.zero_out()

        self.ubirnn = nn.LSTM(input_size=300, hidden_size=embedding_dim,
                              num_layers=self.lstm_layers, batch_first=True,
                              bidirectional=False)
        self.vbirnn = nn.LSTM(input_size=300, hidden_size=embedding_dim,
                              num_layers=self.lstm_layers, batch_first=True,
                              bidirectional=False)

    def init_emb(self):
        """Initialize R and A embeddings"""
        initrange = 0.5 / self.emb_dim
        self.ru_embeddings.weight.data.uniform_(-initrange, initrange)
        # self.ru_embeddings.weight.data[0].zero_()
        self.rv_embeddings.weight.data.uniform_(-0, 0)
        self.au_embeddings.weight.data.uniform_(-initrange, initrange)
        # self.au_embeddings.weight.data[0].zero_()
        self.av_embeddings.weight.data.uniform_(-0, 0)

    def init_hc(self, batch_size):
        h = Variable(
            torch.zeros(self.lstm_layers, batch_size, self.emb_dim))
        c = Variable(
            torch.zeros(self.lstm_layers, batch_size, self.emb_dim))
        if torch.cuda.is_available():
            (h, c) = (h.cuda(), c.cuda())
        return h, c

    def zero_out(self):
        self.ru_embeddings.weight.data[0].zero_()
        self.au_embeddings.weight.data[0].zero_()
        self.rv_embeddings.weight.data[0].zero_()
        self.av_embeddings.weight.data[0].zero_()
