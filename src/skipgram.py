"""
    Skip-gram file

    author: Zeyu Li <zeyuli@ucla.ed> or <zyli@cs.ucla.edu>

    Implemented model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SkipGram(nn.Module):
    """
    Heterogeneous Entity Embedding Based Recommendation
    """
    def __init__(self
                 , embedding_dim
                 , emb_man
                 ):
        super(SkipGram, self).__init__()
        self.emb_dim = embedding_dim
        # self.lambda_=lambda_
        self.embedding_manager = emb_man

    def forward(self, rpos, apos, qinfo):
        emb = self.embedding_manager
        emb.zero_out()
        # R: 0, A: 1, Q: 2
        #print("rpos")
        #print(rpos)
        #print("apos")
        #print(apos)
        embed_ru = emb.ru_embeddings(rpos[0])
        embed_au = emb.au_embeddings(apos[0])

        embed_rv = emb.rv_embeddings(rpos[1])
        embed_av = emb.av_embeddings(apos[1])

        neg_embed_rv = emb.rv_embeddings(rpos[2])
        neg_embed_av = emb.av_embeddings(apos[2])

        quinput, qvinput, qninput = qinfo[:3]
        qulen, qvlen, qnlen = qinfo[3:]

        u_output, _ = emb.ubirnn(quinput, emb.init_hc(quinput.size(0)))
        v_output, _ = emb.vbirnn(qvinput, emb.init_hc(qvinput.size(0)))
        n_output, _ = emb.vbirnn(qninput, emb.init_hc(qninput.size(0)))

        u_pad = Variable(torch.zeros(u_output.size(0), 1, u_output.size(2)))
        v_pad = Variable(torch.zeros(v_output.size(0), 1, v_output.size(2)))
        n_pad = Variable(torch.zeros(n_output.size(0), 1, n_output.size(2)))

        if torch.cuda.is_available():
            u_pad = u_pad.cuda()
            v_pad = v_pad.cuda()
            n_pad = n_pad.cuda()

        u_output = torch.cat((u_pad, u_output), 1)
        v_output = torch.cat((v_pad, v_output), 1)
        n_output = torch.cat((n_pad, n_output), 1)

        # len => len x 1 x self.emb_dim
        qulen = qulen.unsqueeze(1).expand(-1, self.emb_dim).unsqueeze(1)
        qvlen = qvlen.unsqueeze(1).expand(-1, self.emb_dim).unsqueeze(1)
        qnlen = qnlen.unsqueeze(1).expand(-1, self.emb_dim).unsqueeze(1)

        embed_qu = u_output.gather(1, qulen.detach())
        embed_qv = v_output.gather(1, qvlen.detach())
        neg_embed_qv = n_output.gather(1, qnlen.detach())

        embed_u = embed_ru + embed_au + embed_qu.squeeze()
        embed_v = embed_rv + embed_av + embed_qv.squeeze()

        score = torch.mul(embed_u, embed_v)
        score = torch.sum(score)

        log_sigmoid_pos = F.logsigmoid(score).squeeze()
        print("NE loss: positive sample loss {:.6f}"
              .format(log_sigmoid_pos.data[0]))

        neg_embed_v = neg_embed_av + neg_embed_rv + neg_embed_qv.squeeze()
        neg_embed_v = neg_embed_v.view(quinput.size(0), -1, self.emb_dim)

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
        log_sigmoid_neg = F.logsigmoid(-1 * neg_score).squeeze()
        print("NE loss: negative sample loss {:.6f}"
              .format(log_sigmoid_neg.data[0]))

        ne_loss = - (log_sigmoid_pos + log_sigmoid_neg)
        print("NE loss: {:.6f}".format(ne_loss.data[0]))

        # loss = F.sigmoid(ne_loss) + self.lambda_ * F.sigmoid(rank_loss)
        # loss = ne_loss + self.lambda_ * rank_loss
        # loss = ne_loss
        return ne_loss
