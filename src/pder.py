"""

    Personalized Domain Expert Recommendation

    author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>

"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from model import myModel
from data_loader import DataLoader

class PDER:
    def __init__(self, inputfile, vocab_size=1000000, embedding_dim=200,
                 epoch_num=10, batch_size=16, window_size=5,
                 neg_sample_num=10):
        # TODO: all params required
        self.op = DataLoader()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.window_size = window_size
        self.epoch_num = epoch_num
        self.neg_sample_num = neg_sample_num

        pass


    def train(self):
        model = myModel()  #TODO: fill in params
        if torch.cuda.is_available():
            model.cuda()
        optimizer = optim.SGD(model.parameters(), lr=0.2)

        #TODO: change other optmizer

        for epoch in range(self.epoch_num):

            while self.op.process:
                # TODO: Implement generate_batch
                # TODO: what's in the batch
                xx = self.op.generate_batch(self.window_size,
                                         self.batch_size,
                                         self.neg_sample_num)
                xx = Variable(torch.LongTensor(xx))

                if torch.cuda.is_available():
                    xx = xx.cuda()

                optimizer.zero_grad()

                loss = model(xx)

                optimizer.step()

                if not batch_num % 30000:
                    torch.save(model.state_dict(), "path here")

        print("Optimization Finished!")


if __name__ == "__main__":
    my_model = PDER() # TODO: implement here
    my_model.train()
