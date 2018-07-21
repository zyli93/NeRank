"""
    Recommender System file

    author: Zeyu Li <zeyuli@ucla.ed> or <zyli@cs.ucla.edu>

    Implemented model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from collections import OrderedDict

class RecSys(nn.Module):

    def __init__(self, embedding_dim
                 , cnn_channel
                 , embeddings):
        super(RecSys, self).__init__()
        self.emb_dim = embedding_dim
        self.out_channel = cnn_channel
        self.embedding_manager = embeddings

        self.convnet1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, self.out_channel, kernel_size=(1, embedding_dim))),
            ('relu1', nn.ReLU())
            # No max pooling, therefore no dimension reduction
            # ('pool1', nn.MaxPool2d(kernel_size=(3, 1)))
        ]))

        self.convnet2 = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(1, self.out_channel, kernel_size=(2, embedding_dim))),
            ('relu2', nn.ReLU())
            # No max pooling, therefore no dimension reduction
            # ('pool2', nn.MaxPool2d(kernel_size=(2, 1)))
        ]))

        self.convnet3 = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(1, self.out_channel, kernel_size=(3, embedding_dim))),
            ('relu3', nn.ReLU())
        ]))

        self.fc1 = nn.Linear(self.out_channel, 1)
        self.fc2 = nn.Linear(self.out_channel, 1)
        self.fc3 = nn.Linear(self.out_channel, 1)

        # batch x 32 x 6 => batch x 32 x 1 (batch x 32)
        self.fc_new_1 = nn.Linear(6, 1)
        # batch x 32 x 1 => batch x 1 x 1 (batch x 1)
        self.fc_new_2 = nn.Linear(self.out_channel, 1)

    def forward(self, rank):
        """
         === Ranking ===
        """
        emb = self.embedding_manager
        #emb.zero_out()
        #print("emb.ru_embeddings shaoe")
        #print(emb.ru_embeddings.weight.size())
        #print("rank")
        #print(rank)
        emb_rank_r = emb.ru_embeddings(rank[0])
        emb_rank_a = emb.au_embeddings(rank[1])
        emb_rank_acc = emb.au_embeddings(rank[2])
        rank_q, rank_q_len = rank[3], rank[4]

        rank_q_output, _ = emb.ubirnn(rank_q, emb.init_hc(rank_q.size(0)))
        rank_q_pad = Variable(torch.zeros(
            rank_q_output.size(0)
            , 1
            , rank_q_output.size(2))).cuda()
        rank_q_output = torch.cat(
            (rank_q_pad, rank_q_output)
            , 1)

        rank_q_len = rank_q_len.unsqueeze(1).expand(-1, self.emb_dim).unsqueeze(1)

        emb_rank_q = rank_q_output.gather(1, rank_q_len.detach())

        low_rank_mat = torch.stack(
            [emb_rank_r, emb_rank_q.squeeze(), emb_rank_a]
            , dim=1) \
            .unsqueeze(1)
        high_rank_mat = torch.stack(
            [emb_rank_r, emb_rank_q.squeeze(), emb_rank_acc]
            , dim=1) \
            .unsqueeze(1)

        low_score = torch.cat([
            self.convnet1(low_rank_mat)
            , self.convnet2(low_rank_mat)
            , self.convnet3(low_rank_mat)]
            , dim=2).squeeze()
        high_score = torch.cat([
            self.convnet1(high_rank_mat)
            , self.convnet2(high_rank_mat)
            , self.convnet3(high_rank_mat)]
            , dim=2).squeeze()

        low_score = self.fc_new_2(
            self.fc_new_1(low_score.squeeze()).squeeze()).squeeze()
        high_score = self.fc_new_2(
            self.fc_new_1(high_score.squeeze()).squeeze()).squeeze()

        # low_score = self.fc1(self.convnet1(low_rank_mat).view(-1, self.out_channel)) \
        #           + self.fc2(self.convnet2(low_rank_mat).view(-1, self.out_channel)) \
        #           + self.fc3(self.convnet3(low_rank_mat).view(-1, self.out_channel))
        #
        # high_score = self.fc1(self.convnet1(high_rank_mat).view(-1, self.out_channel)) \
        #            + self.fc2(self.convnet2(high_rank_mat).view(-1, self.out_channel)) \
        #            + self.fc3(self.convnet3(high_rank_mat).view(-1, self.out_channel))

        rank_loss = torch.sum(low_score - high_score)
        # rank_loss = F.sigmoid(rank_loss)
        print("Rank loss: {:.6f}".format(rank_loss.data[0]))

        return rank_loss

    def test(self, test_data):
        # test_a, _r, _q all variables
        emb = self.embedding_manager
        test_a, test_r, test_q, test_q_len = test_data
        a_size = test_a.size(0)
        # print(test_a)
        # print(test_r)

        emb_rank_a = emb.au_embeddings(test_a)
        emb_rank_r = emb.ru_embeddings(test_r)

        test_q_output, _ = emb.ubirnn(test_q.unsqueeze(0), emb.init_hc(1))

        ind = Variable(torch.LongTensor([test_q_len])).cuda()
        test_q_target_output = torch.index_select(test_q_output.squeeze(), 0, ind)

        emb_rank_q = test_q_target_output.squeeze() \
            .repeat(a_size).view(a_size, emb.emb_dim)

        emb_rank_mat = torch.stack(
            [emb_rank_r, emb_rank_q, emb_rank_a], dim=1) \
            .unsqueeze(1)
        # emb_rank_mat = emb_rank_mat.unsqueeze(1)
        # score = self.fc1(self.convnet1(emb_rank_mat).view(-1, self.out_channel)) \
        #         + self.fc2(self.convnet2(emb_rank_mat).view(-1, self.out_channel)) \
        #         + self.fc3(self.convnet3(emb_rank_mat).view(-1, self.out_channel))

        score = torch.cat([
            self.convnet1(emb_rank_mat)
            , self.convnet2(emb_rank_mat)
            , self.convnet3(emb_rank_mat)]
            , dim=2).squeeze()

        score = self.fc_new_2(
            self.fc_new_1(score.squeeze()).squeeze()).squeeze()

        ret_score = score.data.squeeze().tolist()
        return ret_score
