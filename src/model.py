"""

    Model file

    author: Zeyu Li <zeyuli@ucla.ed> or <zyli@cs.ucla.edu>

    Implemented model
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
from data_loader import DataLoader

class NeRank(nn.Module):
    """
    Model class

    Heterogeneous Entity Embedding Based Recommendation

    TODO:
        - [ ] Parameters
            - [ ] Check the initialization of embeddings
        - [ ] Efficiency
            - [ ] Run the model in parallel
            - [ ] Move the model to CUDA
        - [ ] Name
    """
    def __init__(self, embedding_dim, dataset):
        super(NeRank, self).__init__()

        # u and v of vector of R, we will use u in the end
        dl = DataLoader(dataset=dataset)
        # TODO: still debating which is better to pass in,
        # TODO:     dl or dataset
        vocab_size = dl.user_count
        self.ru_embeddings = nn.Embedding(vocab_size,
                                         embedding_dim,
                                         sparse=False)
        self.rv_embeddings = nn.Embedding(vocab_size,
                                          embedding_dim,
                                          sparse=False)
        self.au_embeddings = nn.Embedding(vocab_size,
                                          embedding_dim,
                                          sparse=False)
        self.av_embeddings = nn.Embedding(vocab_size,
                                          embedding_dim,
                                          sparse=False)
        self.embedding_dim = embedding_dim
        self.init_emb()

        self.dl = dl  # TODO: adjust here. Get dl from outside

        # TODO: fill in the BiLSTM
        self.birnn = nn.LSTM(input_size=input_size,
                             hidden_size=hiddens_size,
                             num_layers=num_layers,
                             batch_first=True,
                             bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_)




    def init_emb(self):
        """Initialize R and A embeddings"""
        initrange = 0.5 / self.embedding_dim
        self.ru_embeddings.weight.data.uniform_(-initrange, initrange)
        self.rv_embeddings.weight.data.uniform_(-0, 0)
        self.au_embeddings.weight.data.uniform_(-initrange, initrange)
        self.av_embeddings.weight.data.uniform_(-0, 0)


    def forward(self, upos, vpos, npos):
        rupos, qupos, aupos = upos[0], upos[1], upos[2]
        rvpos, qvpos, avpos = vpos[0], vpos[1], vpos[2]

        # TODO: add uid2ind
        # TODO: cannot do such transfer every time it trans

        # TODO: take care of the empty ones
        embed_ru = self.ru_embeddings(rupos)
        embed_au = self.au_embeddings(aupos)

        embed_rv = self.rv_embeddings(rvpos)
        embed_av = self.av_embeddings(avpos)

        embed_qu = # BiRNN TODO: fill in this
        embed_qv = embed_qu # TODO: decide which to use as embed_qu

        embed_u = embed_ru + embed_au + embed_qu
        embed_v = embed_rv + embed_av + embed_qv

        score = torch.mul(embed_u, embed_v)
        score = torch.sum(score)

        log_target = F.logsigmoid(score).squeeze()  # TODO: what is squeeze?

        # TODO: add neg sample embeddings

        # TODO: add ranking things



















