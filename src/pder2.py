"""

    Personalized Domain Expert Recommendation

    author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>

"""

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn

import numpy as np
import time

from model2 import NeRank
from data_loader import DataLoader


class PDER:
    def __init__(self, dataset, embedding_dim, epoch_num,
                 batch_size, window_size, neg_sample_ratio,
                 lstm_layers, include_content, lr, cnn_channel,
                 test_prop, neg_test_ratio, lambda_, prec_k,
                 mp_length, mp_coverage):

        self.dl = DataLoader(dataset=dataset,
                             include_content=include_content,
                             mp_coverage=mp_coverage,
                             mp_length=mp_length)
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.window_size = window_size
        self.epoch_num = epoch_num
        self.neg_sample_ratio = neg_sample_ratio
        self.lstm_layers = lstm_layers
        self.learning_rate = lr

        self.test_prop = test_prop
        self.prec_k = prec_k
        self.neg_test_ratio = neg_test_ratio

        self.model = NeRank(embedding_dim=self.embedding_dim,
                            vocab_size=self.dl.user_count,
                            lstm_layers=self.lstm_layers,
                            cnn_channel=cnn_channel,
                            lambda_=lambda_)  # TODO: fill in params

    def train(self):
        dl = self.dl  # Rename the data loader
        model = self.model
        if torch.cuda.device_count() < 1:
            print("Using {} GPUs".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        if torch.cuda.is_available():  # Check availability of cuda
            model.cuda()

        # TODO: Other learning algorithms
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)

        for epoch in range(self.epoch_num):
            model.train()
            epoch_total_loss = 0
            dl.process = True

            iter = 0

            while dl.process:
                print("Epoch-{}, Iteration-{}".format(epoch, iter), end="")
                if iter == 20:
                    break
                upos, vpos, npos, nsample, aqr, accqr \
                    = dl.generate_batch(
                        window_size=self.window_size,
                        batch_size=self.batch_size,
                        neg_ratio=self.neg_sample_ratio)
                print("\tSample size", nsample)

                """
                In order to totally vectorize the computation, 
                we use following method.
                qupos, qvpos, qnpos are the positions and ids of the text.
                qudiag, qvdiag, qndiag have 1 in cells 
                where the corresponding cell in qupos
                    (qvpos, qnpos) is not zero.
                We use dot product to zero out those undesired columns.

                Create the Variables, only Variables with LongTensor can be
                  sent to nn.Embedding
                """

                # R-u, R-v, and R-n
                rupos = Variable(torch.LongTensor(dl.uid2index(upos[0])))
                rvpos = Variable(torch.LongTensor(dl.uid2index(vpos[0])))
                rnpos = Variable(torch.LongTensor(dl.uid2index(npos[0])))
                rpos = [rupos, rvpos, rnpos]

                # A-u, A-v, and A-n
                aupos = Variable(torch.LongTensor(dl.uid2index(upos[1])))
                avpos = Variable(torch.LongTensor(dl.uid2index(vpos[1])))
                anpos = Variable(torch.LongTensor(dl.uid2index(npos[1])))
                apos = [aupos, avpos, anpos]

                # Q
                # qupos = Variable(torch.LongTensor(upos[2]))
                # qvpos = Variable(torch.LongTensor(vpos[2]))
                # qnpos = Variable(torch.LongTensor(npos[2]))
                # qpos = [qupos, qvpos, qnpos]

                qu_wc = dl.qid2vec_padded(upos[2])
                qv_wc = dl.qid2vec_padded(vpos[2])
                qn_wc = dl.qid2vec_padded(npos[2])

                qulen = dl.qid2vec_len(upos[2])
                qvlen = dl.qid2vec_len(vpos[2])
                qnlen = dl.qid2vec_len(npos[2])

                qu_wc = Variable(torch.FloatTensor(qu_wc).view(-1, dl.PAD_LEN, 300))
                qv_wc = Variable(torch.FloatTensor(qv_wc).view(-1, dl.PAD_LEN, 300))
                qn_wc = Variable(torch.FloatTensor(qn_wc).view(-1, dl.PAD_LEN, 300))
                qulen = Variable(torch.LongTensor(qulen))
                qvlen = Variable(torch.LongTensor(qvlen))
                qnlen = Variable(torch.LongTensor(qnlen))

                qinfo = [qu_wc, qv_wc, qn_wc, qulen, qvlen, qnlen]

                # aqr: R, A, Q

                rank_r = Variable(torch.LongTensor(dl.uid2index(aqr[:, 0])))
                rank_a = Variable(torch.LongTensor(dl.uid2index(aqr[:, 1])))
                rank_acc = Variable(torch.LongTensor(dl.uid2index(accqr)))
                rank_q_wc= dl.qid2vec_padded(aqr[:, 2])
                rank_q_len = dl.qid2vec_len(aqr[:, 2])
                rank_q = Variable(torch.FloatTensor(rank_q_wc).view(-1, dl.PAD_LEN, 300))
                rank_q_len = Variable(torch.LongTensor(rank_q_len))

                # rank_q = Variable(torch.LongTensor(aqr[:, 2]))
                rank = [rank_r, rank_a, rank_acc, rank_q, rank_q_len]

                if torch.cuda.is_available():
                    rpos = [x.cuda() for x in rpos]
                    apos = [x.cuda() for x in apos]
                    # qpos = [x.cuda() for x in qpos]
                    qinfo = [x.cuda() for x in qinfo]
                    rank = [x.cuda() for x in rank]

                optimizer.zero_grad()

                # loss and rank_loss, the later for evaluation
                loss = model(rpos=rpos, apos=apos, qinfo=qinfo,
                             # qpos=qpos,
                             rank=rank, nsample=nsample, dl=dl,
                             test_data=None)

                loss.backward()
                optimizer.step()

                epoch_total_loss += loss.data[0]  # type(loss) = Variable
                iter += 1

                # Save model
                # torch.save(model.state_dict(), "path here")

            print("Epoch-{:d} Loss sum: {}".format(epoch, epoch_total_loss))
            model.eval()
            MRR, pak = self.__validate()
            print("Validation at Epoch-{:d}, \n\tMRR-{:.5f}, Precision@{:d}-{:.5f}"
                  .format(epoch, MRR, self.prec_k, pak))


        print("Optimization Finished!")

    def __validate(self):
        dl = self.dl
        model = self.model
        tbatch = dl.build_test_batch(test_prop=self.test_prop,
                                     test_neg_ratio=self.neg_test_ratio)
        MRR, prec_K = 0, 0
        tbatch_len = len(tbatch)

        # The format of tbatch is:
        #   [aids], rid, qid, accid
        for rid, qid, accid, aid_list in tbatch:
            rank_a = Variable(torch.LongTensor(dl.uid2index(aid_list)))
            rep_rid = [rid] * len(aid_list)
            rank_r = Variable(torch.LongTensor(dl.uid2index(rep_rid)))
            rank_q_len= dl.q2len(qid)
            # TODO: modify rank_r
            rank_q = Variable(torch.FloatTensor(dl.q2emb(qid)))

            if torch.cuda.is_available():
                rank_a = rank_a.cuda()
                rank_r = rank_r.cuda()
                rank_q = rank_q.cuda()
            score = model(rpos=None, apos=None, qinfo=None,
                          rank=None, nsample=None, dl=dl,
                          test_data=[rank_a, rank_r, rank_q, rank_q_len], train=False)
            RR, prec = dl.perform_metric(aid_list, score.tolist(),
                                         accid, self.prec_k)
            MRR += RR
            prec_K += prec

        MRR, prec_K= MRR / len(tbatch_len), prec_K / len(tbatch_len)
        return MRR, prec_K


    def test(self):
        print("Testing under construction.")
        pass
        # TODO: implement here


if __name__ == "__main__":
    pder = PDER() # TODO: implement here
    pder.train()
    pder.test()
