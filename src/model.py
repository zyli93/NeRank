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
        self.dl = DataLoader(dataset=dataset)
        # TODO: still debating which is better to pass in, dl or dataset
        vocab_size = self.dl.user_count
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
        # TODO: it is very likely that later on we will move the
        # TODO:     transformation to outer structure.

        dl = self.dl

        rupos, qupos, aupos = upos[0], upos[1], upos[2]
        rvpos, qvpos, avpos = vpos[0], vpos[1], vpos[2]
        rnpos, qnpos, anpos = npos[0], npos[1], npos[2]

        batch_size = rupos.shape[0]
        # TODO: came up with better way to get shape

        """
        === Network Embedding Part ===
        """
        # uid representation to user index representation
        rupos, rvpos = dl.uid2index(rupos), dl.uid2index(rvpos)
        aupos, avpos = dl.uid2index(aupos), dl.uid2index(avpos)
        rnpos, anpos = dl.uid2index(rnpos), dl.uid2index(anpos)
        # TODO: cannot do such transfer every time it trans

        # TODO: take care of the empty ones
        embed_ru = self.ru_embeddings(rupos)
        embed_au = self.au_embeddings(aupos)

        embed_rv = self.rv_embeddings(rvpos)
        embed_av = self.av_embeddings(avpos)

        # Specifically handle the text
        embed_qu = torch.LongTensor(self.embedding_dim, batch_size).zero_()
        for i, qid in enumerate(qupos):
            if qid:
                x = torch.LongTensor(dl.qid2vecs(qid))
                # TODO: check correctness, whether to add view() or squeeze()?
                embed_qu[i] = self.birnn(x, self.hidden)  # TODO: what is hidden?

        embed_qv = embed_qu # TODO: decide which to use as embed_qu

        embed_u = embed_ru + embed_au + embed_qu
        embed_v = embed_rv + embed_av + embed_qv

        score = torch.mul(embed_u, embed_v)
        score = torch.sum(score)

        log_target = F.logsigmoid(score).squeeze()  # TODO: what is squeeze?

        # TODO: add neg sample embeddings
        neg_batch_size = rnpos.shape[0]
        neg_embed_rv = self.rv_embeddings(rnpos)
        neg_embed_av = self.av_embeddings(anpos)
        neg_embed_qv = torch.LongTensor(self.embedding_dim, neg_batch_size).zero_()
        for i, qid in enumerate(qnpos):
            if qid:
                x = torch.LongTensor(dl.qid2vecs(qid))
                neg_embed_qv[i] = self.birnn(x, self.hidden)

        neg_embed_v = neg_embed_av + neg_embed_rv + neg_embed_qv
        neg_score = torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze()
        # TODO: why here unsqueeze and squeeze?
        neg_score = torch.sum(neg_score)
        sum_log_sampled = F.logsigmoid(-1 * neg_score).squeeze()

        loss = log_target + sum_log_sampled



        # TODO: add ranking things


