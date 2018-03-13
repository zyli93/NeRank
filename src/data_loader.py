"""
    Data Loader

    the class of loading data

    author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>

"""

import numpy as np
import os

class DataLoader(object):
    def __init__(self, metapath_file, question_file, vocab_size):
        self.metapath_file = metapath_file
        self.question_file = question_file
        self.vocab_size = vocab_size

    def generate_batch(self, window_size, batch_size, count):
        data = self.train_data
        global data_index

        # the batch got is actually a word pair
