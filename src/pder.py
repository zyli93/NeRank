"""

    Personalized Domain Expert Recommendation

    author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>

"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import time

from model import NeRank
from data_loader import DataLoader


class PDER:
    def __init__(self, dataset,
                 vocab_size=1000000, embedding_dim=200,
                 epoch_num=10, batch_size=16,
                 window_size=5, neg_sample_ratio=10):

        self.dl = DataLoader(dataset=dataset) # TODO: all params required
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.window_size = window_size
        self.epoch_num = epoch_num
        self.neg_sample_ration = neg_sample_ratio

        pass


    def train(self):

        dl = self.dl  # Rename the data loader
        model = NeRank()  # TODO: fill in params
        if torch.cuda.is_available():  # Check availability of cuda
            model.cuda()

        optimizer = optim.SGD(model.parameters(), lr=0.2)  # TODO: to other optimizer

        for epoch in range(self.epoch_num):
            start = time.time()
            dl.process = True

            while dl.process:
                # TODO: Here comes the get data and preprocessing
                # TODO: Implement generate_batch
                # TODO: what's in the batch
                upos, vpos, npos = dl.generate_batch(
                    window_size=self.window_size,
                    batch_size=self.batch_size,
                    neg_ratio=self.neg_sample_ration)

                # UID representation to user index representation
                # TODO: npos don't apply because of the dimension
                # TODO: Seems that diag doesn't work with 2+ dim matrices
                rupos, rvpos, rnpos = dl.uid2index(upos[0]), \
                                      dl.uid2index(vpos[0]), \
                                      dl.uid2index(npos[0])
                aupos, avpos, anpos = dl.uid2index(upos[1]), \
                                      dl.uid2index(vpos[1]), \
                                      dl.uid2index(npos[1])
                qupos, qvpos, qnpos = upos[2], vpos[2], npos[2]
                """
                In order to totally vectorize the computation, we use following method.
                qupos, qvpos, qnpos are the positions and ids of the text.
                qudiag, qvdiag, qndiag have 1 in cells where the corresponding cell in qupos
                    (qvpos, qnpos) is not zero.
                We use dot product to zero out those undesired columns.
                """
                qudiag, qvdiag, qndiag = [np.diag(np.where(loc > 0, 1, 0))
                                          for loc in [qupos, qvpos, qnpos]]  # TODO: fix qnpos

                # qupos, qvpos = dl.uid2index(upos[2]), dl.uid2index(vpos[2])


                # Create the Variables, only Variables with LongTensor can be
                #   sent to nn.Embedding

                # R-u and R-v
                rupos = Variable(torch.LongTensor(rupos))
                rvpos = Variable(torch.LongTensor(rvpos))

                # A-u and A-v
                aupos = Variable(torch.LongTensor(aupos))
                avpos = Variable(torch.LongTensor(avpos))

                # Q ???

                # Negative Samples of R and A
                rnpos = Variable(torch.LongTensor(rnpos))
                anpos = Variable(torch.LongTensor(anpos))


                variables = [
                    rupos, rvpos, rnpos,
                    aupos, avpos, anpos
                ]

                if torch.cuda.is_available():
                    rupos = rupos.cuda()
                    rvpos = rvpos.cuda()
                    rnpos = rnpos.cuda()
                    aupos = aupos.cuda()
                    avpos = avpos.cuda()
                    anpos = anpos.cuda()
                    # TODO: fill in Q


                optimizer.zero_grad()

                loss = model(upos, vpos, npos, )

                optimizer.step()

                if not batch_num % 30000:
                    torch.save(model.state_dict(), "path here")

        print("Optimization Finished!")

    def evaluate(self):
        pass
        # TODO: implement here


if __name__ == "__main__":
    pder = PDER() # TODO: implement here
    pder.train()
