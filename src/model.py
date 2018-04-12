"""

    Model file

    author: Zeyu Li <zeyuli@ucla.ed> or <zyli@cs.ucla.edu>

    Implemented model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from data_loader import DataLoader

from collections import OrderedDict


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
    def __init__(self, embedding_dim, vocab_size, lstm_layers):
        super(NeRank, self).__init__()

        # u and v of vector of R, we will use u in the end
        self.emb_dim = embedding_dim

        self.lstm_layers = lstm_layers
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
        self.hc = self.init_hc()


        self.ubirnn = nn.LSTM(input_size=embedding_dim,
                              hidden_size=embedding_dim,
                              num_layers=self.lstm_layers,
                              batch_first=True,
                              bidirectional=True)
        self.vbirnn = nn.LSTM(input_size=embedding_dim,
                              hidden_size=embedding_dim,
                              num_layers=self.lstm_layers,
                              batch_first=True,
                              bidirectional=True)

        # TODO: set up the size of the out_channel
        out_cha = 32
        self.convnet1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, out_cha, kernel_size=(1, embedding_dim))),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(kernel_size=(3, 1)))
        ]))

        self.convnet2 = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(2, out_cha, kernel_size=(2, embedding_dim))),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool2d(kernel_size=(2, 1)))
        ]))

        self.convnet3 = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(3, out_cha, kernel_size=(3, embedding_dim))),
            ('relu3', nn.ReLU())
        ]))

        self.fc1 = nn.Linear(out_cha, 1)
        self.fc2 = nn.Linear(out_cha, 1)
        self.fc3 = nn.Linear(out_cha, 1)

    def init_emb(self):
        """Initialize R and A embeddings"""
        initrange = 0.5 / self.embedding_dim
        self.ru_embeddings.weight.data.uniform_(-initrange, initrange)
        self.ru_embeddings.weight.data[0].zero_()
        self.rv_embeddings.weight.data.uniform_(-0, 0)
        self.au_embeddings.weight.data.uniform_(-initrange, initrange)
        self.ru_embeddings.weight.data[0].zero_()
        self.av_embeddings.weight.data.uniform_(-0, 0)

    def init_hc(self):
        h = Variable(torch.zeros(self.lstm_layers * 2, 1, self.embedding_dim))
        c = Variable(torch.zeros(self.lstm_layers * 2, 1, self.embedding_dim))
        if torch.cuda.is_available():
            (h, c) = (h.cuda(), c.cuda())
        return h, c

    def forward(self, rupos, rvpos, rnpos, aupos, avpos, anpos,
                      quloc, qvloc, qnloc, quemb, qvemb, qnemb,
                      nsample,
                      rank_a, rank_acc, rank_r, rank_q_emb):
        """
        forward algorithm for NeRank,
        quloc, qvloc, qnloc are locations in a vector of u, v,
            and negative samples where there is a question text.
        quemb, qvemb, qnemb are the word embedding piles of that
        """

        """
            === Network Embedding Part ===
        """
        embed_ru = self.ru_embeddings(rupos)
        embed_au = self.au_embeddings(aupos)
        embed_rv = self.rv_embeddings(rvpos)
        embed_av = self.av_embeddings(avpos)

        # quemb here is just the concatenation of word vectors
        #   after this step, everything is equal length

        _, (lstm_quemb_hidden, _) = self.ubirnn(quemb.unsqueeze(1), self.hc)

        embed_qu = torch.matmul(torch.diag(quloc), lstm_quemb_hidden)

        _, (lstm_qvemb_hidden, _) = self.vbirnn(qvemb.unsqueeze(1), self.hc)
        embed_qv = torch.matmul(torch.diag(qvloc), lstm_qvemb_hidden)

        embed_u = embed_ru + embed_au + embed_qu
        embed_v = embed_rv + embed_av + embed_qv

        score = torch.mul(embed_u, embed_v)
        score = torch.sum(score)

        log_target = F.logsigmoid(score).squeeze()

        neg_embed_rv = self.rv_embeddings(rnpos)
        neg_embed_av = self.av_embeddings(anpos)

        _, (lstm_qnemb_hidden, _) = self.vbirnn(qnemb.unsqueeze(1), self.hc)
        neg_embed_qv = torch.matmul(torch.diag(qnloc), lstm_qnemb_hidden)

        neg_embed_v = neg_embed_av + neg_embed_rv + neg_embed_qv
        neg_embed_v = neg_embed_v.view(nsample, self.emb_dim, -1)

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

        ne_loss = log_target + sum_log_sampled

        """
            === Ranking ===
        """

        emb_rank_acc = self.au_embeddings(rank_acc)
        emb_rank_a = self.au_embeddings(rank_a)
        emb_rank_r = self.ru_embeddings(rank_r)
        emb_rank_q = self.ubirnn(rank_q_emb)

        low_rank_mat = torch.stack([emb_rank_r, emb_rank_q, emb_rank_a],
                                   dim=1)
        high_rank_mat = torch.stack([emb_rank_r, emb_rank_q, emb_rank_acc],
                                    dim=1)

        low_score = self.fc1(self.convnet1(low_rank_mat)) \
                  + self.fc2(self.convnet2(low_rank_mat)) \
                  + self.fc3(self.convnet3(low_rank_mat))

        high_score = self.fc3(self.convnet1(high_rank_mat)) \
                   + self.fc2(self.convnet2(high_rank_mat)) \
                   + self.fc3(self.convnet3(high_rank_mat))

        rank_loss = low_score - high_score

        loss = ne_loss + rank_loss
        return loss
