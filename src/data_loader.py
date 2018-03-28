"""
    Data Loader

    the class of loading data

    author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>

"""

import numpy as np
import pandas as pd
import os

class DataLoader():
    def __init__(self, question_file, vocab_size, dataset):
        self.dataset = dataset
        self.mpfile = os.getcwd() + "/metapath/{}.txt".format(self.dataset)
        self.Qfile = question_file
        self.vocab_size = vocab_size

        data = self.read_data()

        # TODO: DECIDE subsampling or all data
        if True:
            self.train_data = self.subsampling(data)
        else:
            self.train_data = data

        self.sample_table = self.init_sample_table()

    def read_data(self):
        with open(self.mpfile, "r") as fin:
            lines = fin.readlines()
            data = [line.strip().split(" ") for line in lines]
            return data

    def generate_batch(self, window_size, batch_size, count):
        data = self.train_data
        global data_index

        # the batch got is actually a word pair


if __name__ == "__main__":
    test = DataLoader(question_file="x", vocab_size=1, dataset="3dprinting")

