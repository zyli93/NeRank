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
        if torch.cuda.device_count() < 1:
            print("Using {} GPUs".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        if torch.cuda.is_available():  # Check availability of cuda
            model.cuda()

        optimizer = optim.SGD(model.parameters(), lr=0.2)  # TODO: to other optimizer

        for epoch in range(self.epoch_num):
            start = time.time()
            dl.process = True

            iter = 0

            while dl.process:
                print("Epoch-{}, Iteration-{}".format(epoch, iter), end="")
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
                qupos = Variable(torch.LongTensor(upos[2]))
                qvpos = Variable(torch.LongTensor(vpos[2]))
                qnpos = Variable(torch.LongTensor(npos[2]))
                qpos = [qupos, qvpos, qnpos]

                # aqr: R, A, Q

                rank_r = Variable(torch.LongTensor(dl.uid2index(aqr[:, 0])))
                rank_a = Variable(torch.LongTensor(dl.uid2index(aqr[:, 1])))
                rank_acc = Variable(torch.LongTensor(dl.uid2index(accqr)))
                rank_q = Variable(torch.LongTensor(aqr[:, 2]))
                rank = [rank_r, rank_a, rank_acc, rank_q]

                # === ALL INPUT PROCESS ARE BEFORE THIS LINE ===
                if torch.cuda.is_available():
                    rpos = [x.cuda() for x in rpos]
                    apos = [x.cuda() for x in apos]
                    qpos = [x.cuda() for x in qpos]
                    rank = [x.cuda() for x in rank]

                optimizer.zero_grad()

                loss = model(rpos=rpos, apos=apos, qpos=qpos,
                             rank=rank, nsample=nsample, dl=dl)

                loss.backward()

                optimizer.step()

                iter += 1

                # Save model
                # torch.save(model.state_dict(), "path here")

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
