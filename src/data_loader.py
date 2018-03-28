"""
    Data Loader

    the class of loading data

    author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>

    # TODO: do I really have to do this?
    # TODO: how to subsampling, how to negtive sampling, that it

"""

import numpy as np
import pandas as pd
import os
import collections
import functools

class DataLoader():
    def __init__(self, question_file, vocab_size, dataset):
        self.dataset = dataset
        self.mpfile = os.getcwd() + "/metapath/{}.txt".format(self.dataset)
        self.Qfile = question_file
        self.vocab_size = vocab_size

        data = self.read_data()
        self.count = self.count_dataset(data)
        self.sample_table = self.init_sample_table()

        # TODO: DECIDE subsampling or all data
        if True:
            self.train_data = self.subsampling(data)
        else:
            self.train_data = data


    def read_data(self):
        with open(self.mpfile, "r") as fin:
            lines = fin.readlines()
            data = [line.strip().split(" ") for line in lines]
            return data

    def count_dataset(self, data):
        count_dict = {}
        for path in data:
            for entity in path:
                count_dict[entity] = count_dict.get(entity, 0) + 1

        # keys: entity id, values: count
        count = list(zip(count_dict.keys(), count_dict.values()))
        count.sort(key=lambda x:x[1], reverse=True)
        return count

    def init_sample_table(self):
        count = [ele[1] for ele in self.count]
        pow_freq = np.array(count) ** 0.75
        ratio = pow_freq / sum(pow_freq)
        table_size = 1e8 # TODO: what is this???
        count = np.round(ratio * table_size)
        sample_table = []
        for index, x in enumerate(count):
            pass #TODO: implement this


        pass

    def subsampling(self, data):
        # TODO: found an tutorial on this
        pass

    def generate_batch(self, window_size, batch_size, count):
        data = self.train_data
        global data_index
        span = 2 * window_size + 1
        context = np.ndarray(shape=(batch_size, 2 * window_size), dtype=str)
            # TODO: check if the dtype is correct, can it be optimized?
        labels = np.ndarray(shape=(batch_size), dtype=str)
        pos_pair = []


        # the batch got is actually a word pair

    def sen2vecs(self, sentence):
        #


if __name__ == "__main__":
    test = DataLoader(question_file="x", vocab_size=1, dataset="3dprinting")

