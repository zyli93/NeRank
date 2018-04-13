"""

    Personalized Domain Expert Recommendation

    author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>

"""

import torch
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
import time

from model import NeRank
from data_loader import DataLoader


class PDER:
    def __init__(self, dataset, embedding_dim, epoch_num,
                 batch_size, window_size, neg_sample_ratio,
                 lstm_layers, include_content):
        self.dl = DataLoader(dataset=dataset,
                             include_content=include_content)
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.window_size = window_size
        self.epoch_num = epoch_num
        self.neg_sample_ratio = neg_sample_ratio
        self.lstm_layers = lstm_layers


    def train(self):

        dl = self.dl  # Rename the data loader
        model = NeRank(embedding_dim=self.embedding_dim,
                       vocab_size=dl.user_count,
                       lstm_layers=self.lstm_layers)  # TODO: fill in params
        if torch.cuda.is_available():  # Check availability of cuda
            model.cuda()

        optimizer = optim.SGD(model.parameters(), lr=0.2)  # TODO: to other optimizer

        for epoch in range(self.epoch_num):
            start = time.time()
            dl.process = True

            while dl.process:
                upos, vpos, npos, nsample, aqr, accqr \
                    = dl.generate_batch(
                        window_size=self.window_size,
                        batch_size=self.batch_size,
                        neg_ratio=self.neg_sample_ratio)

                # UID representation to user index representation
                rupos, rvpos, rnpos = dl.uid2index(upos[0]), \
                                      dl.uid2index(vpos[0]), \
                                      dl.uid2index(npos[0])
                aupos, avpos, anpos = dl.uid2index(upos[1]), \
                                      dl.uid2index(vpos[1]), \
                                      dl.uid2index(npos[1])
                qupos, qvpos, qnpos = upos[2], vpos[2], npos[2]

                """
                In order to totally vectorize the computation, 
                we use following method.
                qupos, qvpos, qnpos are the positions and ids of the text.
                qudiag, qvdiag, qndiag have 1 in cells 
                where the corresponding cell in qupos
                    (qvpos, qnpos) is not zero.
                We use dot product to zero out those undesired columns.
                """

                # Create the Variables, only Variables with LongTensor can be
                #   sent to nn.Embedding

                # R-u and R-v
                rupos = Variable(torch.LongTensor(rupos))
                rvpos = Variable(torch.LongTensor(rvpos))

                # A-u and A-v
                aupos = Variable(torch.LongTensor(aupos))
                avpos = Variable(torch.LongTensor(avpos))

                # Negative Samples of R and A
                rnpos = Variable(torch.LongTensor(rnpos))
                anpos = Variable(torch.LongTensor(anpos))

                # Q
                quloc = Variable(torch.LongTensor(np.where(qupos > 0, 1, 0)))
                qvloc = Variable(torch.LongTensor(np.where(qvpos > 0, 1, 0)))
                qnloc = Variable(torch.LongTensor(np.where(qnpos > 0, 1, 0)))

                quemb = Variable(torch.FloatTensor(dl.qid2vec(qupos)))
                qvemb = Variable(torch.FloatTensor(dl.qid2vec(qvpos)))
                qnemb = Variable(torch.FloatTensor(dl.qid2vec(qnpos)))

                # aqr: R, A, Q
                rank_r, rank_a = dl.uid2index(aqr[:, 0]), \
                                 dl.uid2index(aqr[:, 1])
                rank_acc = dl.uid2index(accqr)
                rank_q = aqr[:, 2]

                rank_a_var = Variable(torch.LongTensor(rank_a))
                rank_acc_var = Variable(torch.LongTensor(rank_acc))
                rank_r_var = Variable(torch.LongTensor(rank_r))
                rank_q_emb = Variable(torch.FloatTensor(dl.qid2vec(rank_q)))

                # === ALL INPUT PROCESS ARE BEFORE THIS LINE ===
                if torch.cuda.is_available():
                    rupos = rupos.cuda()
                    rvpos = rvpos.cuda()
                    rnpos = rnpos.cuda()
                    aupos = aupos.cuda()
                    avpos = avpos.cuda()
                    anpos = anpos.cuda()
                    quloc = quloc.cuda()
                    qvloc = qvloc.cuda()
                    qnloc = qnloc.cuda()
                    quemb = quemb.cuda()
                    qvemb = qvemb.cuda()
                    qnemb = qnemb.cuda()

                    rank_a_var = rank_a_var.cuda()
                    rank_acc_var = rank_acc_var.cuda()
                    rank_r_var = rank_r_var.cuda()
                    rank_q_emb = rank_q_emb.cuda()

                optimizer.zero_grad()

                loss = model(rupos, rvpos, rnpos,
                             aupos, avpos, anpos,
                             quloc, qvloc, qnloc,
                             quemb, qvemb, qnemb,
                             nsample,
                             rank_a_var,
                             rank_acc_var,
                             rank_r_var,
                             rank_q_emb)  # TODO: fill in this

                optimizer.step()

                if not batch_num % 30000:
                    torch.save(model.state_dict(), "path here")

            # TODO: evaluate

        self.evaluate()
        print("Optimization Finished!")

    def evaluate(self):
        print("Evaluation under construction.")
        pass
        # TODO: implement here

    def test(self):
        print("Testing under construction.")
        pass
        # TODO: implement here


if __name__ == "__main__":
    pder = PDER() # TODO: implement here
    pder.train()
