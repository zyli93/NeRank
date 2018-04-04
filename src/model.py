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
    def __init__(self, vocab_size, embedding_dim):
        super(NeRank, self).__init__()

        # u and v of vector of R, we will use u in the end
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

        dl = DataLoader(dataset="3dprinting")




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

        # TODO: take care of the empty ones
        embed_ru = self.ru_embeddings(rupos)
        embed_au = self.au_embeddings(aupos)

        embed_rv = self.rv_embeddings(rvpos)
        embed_av = self.av_embeddings(avpos)











