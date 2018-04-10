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
        self.ru_embeddings.weight.data[0].zero_()
        self.rv_embeddings.weight.data.uniform_(-0, 0)
        self.au_embeddings.weight.data.uniform_(-initrange, initrange)
        self.ru_embeddings.weight.data[0].zero_()
        self.av_embeddings.weight.data.uniform_(-0, 0)


    def forward(self, rupos, rvpos, rnpos,
                      aupos, avpos, anpos,
                      quloc, qvloc, qnloc,
                      quemb, qvemb, qnemb):
        """
        forward algorithm for NeRank,
        quloc, qvloc, qnloc are locations in a vector of u, v,
            and negative samples where there is a question text.
        quemb, qvemb, qnemb are the word embedding piles of that
        """

        dl = self.dl

        """
                === Network Embedding Part ===
        """

        embed_ru = self.ru_embeddings(rupos)
        embed_au = self.au_embeddings(aupos)

        embed_rv = self.rv_embeddings(rvpos)
        embed_av = self.av_embeddings(avpos)


        # quemb here is just the concatenation of word vectors
        # after this step, everything is equal length
        embed_qu = self.birnn(quemb)  # TODO: the input format of BiLSTM
        embed_qv = self.birnn(qvemb)
        embed_qn = self.birnn(qnemb)

        embed_qu = torch.


        #=================================================

        embed_u = embed_ru + embed_au + embed_qu
        embed_v = embed_rv + embed_av + embed_qv

        score = torch.mul(embed_u, embed_v)
        score = torch.sum(score)

        log_target = F.logsigmoid(score).squeeze()

        neg_batch_size = rnpos.shape[0]
        neg_embed_rv = self.rv_embeddings(rnpos)
        neg_embed_av = self.av_embeddings(anpos)
        neg_embed_qv = torch.LongTensor(self.embedding_dim,
                                        neg_batch_size).zero_()
        for i, qid in enumerate(qnpos):
            if qid:
                x = torch.LongTensor(dl.qid2vecs(qid))
                neg_embed_qv[i] = self.birnn(x, self.hidden)

        neg_embed_v = neg_embed_av + neg_embed_rv + neg_embed_qv
        """
        Some notes around here.
        * unsqueeze(): add 1 dim in certain position
        * squeeze():   remove all 1 dims. E.g. (4x1x2x4x1) -> (4x2x4)
        * Explain the dimension:
            bmm: batch matrix-matrix product.
                batch1 - b x n x m
                batch2 - b x m x p
                return - b x n x p
            Here:
                neg_embed_v - 2*batch_size*window_size x count x emb_dim
                embed_u     - 2*batch_size*window_size x emb_dim
                embed_u.unsqueeze(2)
                            - 2*batch_size*window_size x emb_dim x 1
                bmm(.,.)    - 2*batch_size*window_size x count x 1
                bmm(.,.).squeeze()
                            - 2*batch_size*window_size x count
        * Input & Output of nn.Embeddings:
            In : LongTensor(N,M)
            Out: (N, W, embedding_dim)
        """
        neg_score = torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze()
        neg_score = torch.sum(neg_score)
        sum_log_sampled = F.logsigmoid(-1 * neg_score).squeeze()

        loss = log_target + sum_log_sampled





        # TODO: add ranking things
        """
            === Learning to Rank Part ===
        """
        # TODO:
        #



