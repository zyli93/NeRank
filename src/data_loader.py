"""
    Data Loader

    the class of loading data

    author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>

    # TODO: how to subsampling, how to negtive sampling, that it

"""

import numpy as np
import os, sys

import gensim


class DataLoader():
    def __init__(self, vocab_size, dataset):
        print("Initializing data_loader ...", end=" ")
        self.dataset = dataset
        self.mpfile = os.getcwd() + "/metapath/"+ self.dataset +".txt"
        self.datadir = os.getcwd() + "/data/parsed/" + self.dataset + "/"
        self.vocab_size = vocab_size

        data = self.read_data()
        self.train_data = data

        self.count = self.count_dataset(data)
        self.sample_table = self.init_sample_table()
        self.w2vmodel = self.load_word2vec()  # **Time Consuming!**
        self.qid2sen = self.load_questions()

        print("Done!")


    def read_data(self):
        """
        Read metapath dataset

        Return:
            data  -  the metapath dataset
        """
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

        for i in range(len(self.count)):
            sample_table += [self.count[i][0]] * count[i]


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

    def qid2vecs(self, qid):
        """
        Given Qid, return the concatenated word vectors

        Args:
            qid  -  the qid

        Return:
            qvec  -  the vector of the question
        """
        question = self.qid2sen[qid]
        qvec = self.sen2vecs(sentence=question)
        return qvec

    def sen2vecs(self, sentence):
        """
        Convert sentence to concatenate of word-vectors

        Arg:
            sentence  -  str, the sentence to convert

        Return:
            (np.array)  -  the array contains the word vectors
        """
        sentence = sentence.split(" ")
        vectors = [self.w2vmodel[w] for w in sentence]
        return len(sentence), np.array(vectors)

    def load_word2vec(self):
        """
        Loading word2vec model, return the model

        Return:
            model  -  dictionary, ("word", [word-vector]),
                      loaded word2vec model
        """
        PATH = os.getcwd() + "/word2vec_model/" +\
                "GoogleNews-vectors-negative300.bin"
        model = gensim.models.KeyedVectors.load_word2vec_format(
            fname=PATH, binary=True)
        return model

    def load_questions(self):
        """
        Load question from dataset, "title" + "content",
            construct the qid2sen dictionary

        Return:
            qid2sen  -  the qid:question dictionary
        """

        qcfile = self.datadir + "Q_content_nsw.txt"
        qtfile = self.datadir + "Q_title_nsw.txt"

        qid2sen = {}

        with open(qtfile, "r") as fin_t:
            lines = fin_t.readlines()
            for line in lines:
                id, title = line.split(" ", 1)
                qid2sen[id] = title

        with open(qcfile, "r") as fin_c:
            lines = fin_c.readlines()
            for line in lines:
                id, content = line.split(" ", 1)
                qid2sen[id] += " " + content

        return qid2sen


if __name__ == "__main__":
    test = DataLoader(vocab_size=1, dataset="3dprinting")
    test.load_word2vec()

