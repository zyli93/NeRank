"""
    Data Loader

    the class of loading data

    author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>

"""

import numpy as np
import os, sys

import gensim
import random

from collections import Counter

data_index = 0
test_index = 0

class DataLoader():
    def __init__(self, dataset, ID,
                 include_content, coverage, length, answer_sample_ratio):
        print("Initializing data_loader ...")
        self.ANS_SAMPLE_SIZE = 5
        self.PAD_LEN = 256
        self.id = ID
        self.dataset = dataset
        self.include_content = include_content
        self.process = True
        self.answer_sample_ratio = answer_sample_ratio

        self.corpus_path =\
            os.getcwd() + "/corpus/" + "{}_{}_{}.txt".format(
                self.dataset, str(coverage), str(length))

        self.mpwalks_path =\
            os.getcwd() + "/metapath/" + "{}_{}_{}.txt".format(
                self.dataset, str(coverage), str(length))

        self.DATA_DIR = os.getcwd() + "/data/parsed/{}/".format(self.dataset)

        print("\tLoading dataset ..." + self.corpus_path)
        self.data = self.__read_data()

        print("\tCounting dataset ...")
        self.count = self.__count_dataset()

        print("\tInitializing sample table ...")
        self.sample_table = self.__init_sample_table()

        print("\tLoading word2vec model ...")
        self.w2vmodel = self.__load_word2vec()  # **time consuming!**

        print("\tLoading questions text ...")
        self.question_text = self.__load_question_text()

        print("\tCreating user-index mapping ...")
        self.uid2ind, self.ind2uid = {}, {}
        self.user_count = self.__create_uid_index()

        print("\tLoading rqa ...")
        self.q2r, self.q2acc, self.q2a = {}, {}, {}
        self.all_aid = []
        self.__load_rqa()

        print("\tCreating qid embeddings map ...")
        self.qid2emb, self.qid2len = {}, {}
        self.__get_question_embeddings()

        print("\tLoading test sets ...")
        self.testset = self.__load_test()

        print("Done - Data Loader!")

    def __read_data(self):
        """
        read metapath dataset,
            load the dataset into data

        return:
            data  -  the metapath dataset
        """
        with open(self.corpus_path, "r") as fin:
            lines = fin.readlines()
            data = [line.strip().split(" ") for line in lines]
            return data

    def __count_dataset(self):
        """
        read dataset and count the frequency

        args:
            data  -  the list of meta-pathes.
        returns:
            count  - the sorted list of
        """
        count_dict = {}
        counter = Counter()
        with open(self.mpwalks_path, "r") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.strip().split(" ")
                counter.update(line)
            # line = fin.readline()
            # path = line.strip().split(" ")
            # for entity in path:
                # count_dict[entity] = count_dict.get(entity, 0) + 1

        # keys: entity id, values: count
        #count = list(zip(count_dict.keys(), count_dict.values()))
        #count.sort(key=lambda x: x[1], reverse=True)
        return counter.most_common()

    def __init_sample_table(self):
        """
        create sample tables by p()^(3/4)

        return:
            (sample_table)  -  the created sample table
        """
        count = [ele[1] for ele in self.count]
        pow_freq = np.array(count) ** 0.75
        ratio = pow_freq / sum(pow_freq)
        table_size = 2e7 # todo: what is this???
        count = np.round(ratio * table_size).astype(np.int64)
        sample_table = []

        for i in range(len(self.count)):
            sample_table += [self.count[i][0]] * count[i]
        return np.array(sample_table)

    def get_train_batch(self, batch_size, neg_ratio):
        """
        get batch from the meta paths for the training of skip-gram
            model.

        args:
            window_size  -  the size of sliding window. in following case,
                            the window size is 2. "_ _ [ ] _ _"
            batch_size   -  how many meta-paths compose a batch
            neg_ratio    -  the ratio of negative samples w.r.t.
                            the positive samples
        return:
            * all these vecs are in form of "[eny]_[id]" format
            upos         -  the u vector positions (1d tensor)
            vpos         -  the v vector positions (1d tensor)
            npos         -  the negative samples positions (2d tensor)
        """
        data = self.data
        global data_index

        if batch_size + data_index < len(data):
            batch_pairs = data[data_index: data_index + batch_size]
            data_index += batch_size
        else:
            batch_pairs = data[data_index:]
            data_index = 0
            self.process = False

        u, v = zip(*batch_pairs)
        upos = self.__separate_entity(u)
        vpos = self.__separate_entity(v)

        neg_samples = np.random.choice(
            self.sample_table,
            size=int(len(batch_pairs) * neg_ratio))
        npos = self.__separate_entity(neg_samples)
        aqr, accqr = self.get_answer_sample(upos, self.ANS_SAMPLE_SIZE)
        return upos, vpos, npos, aqr, accqr

    def get_test_batch(self, test_prop):
        """
        Build a batch for test

        Args:
            test_prop      -  the ratio of data fed to test,
                               if None, use all batch
            test_neg_ratio  -  the ratio of negative test instances,
                               expected to be an integer

        Returns:
            A list of test data, formatted as follows:
                trank_a - a list of answers (vector)
                rid - the raiser ID (scalar)
                qid - the question ID (scalar)
                accaid - the accepted answer owner ID (scaler)
        """
        # total = self.testset.shape[0]
        total = len(self.testset)
        if test_prop:
            batch_size = int(total * test_prop)
            batch = random.sample(self.testset, batch_size)
        else:
            batch = self.testset
        return batch

    def get_answer_sample(self, upos, sample_size):
        """
        This method is for Ranking CNN

        Args:
            upos  -  Label entity column
            vpos  -  Context entity column
        Return:
            aqr   -  three cols list:
                     A of upos, Q of vpos, R of this Q
            acc   -  one col list:
                     the Accepted answer in the corresponding pos

        TODO:
            For now, we only look at AQ pair. We can also implement
            AR pair to list of Q's and sample some Q to construct tuples.
            But those are left for future implementation.
        WHY:
        In upos and vpos, find all following pairs:
            1 - "A-Q"
            2 - "A-R" (Not implemented)
        construct:
            "A-R-Q", "A*-R-Q"
        """
        length = upos.shape[1]

        # R: 0, A: 1, Q: 2
        datalist = []
        acclist = []

        for i in range(length):
            if upos[2][i]:
                qid = upos[2][i]
                aid_samples = self.q2a[qid]
                accaid = self.q2acc[qid]
                if len(aid_samples) < sample_size:
                    more_ans = np.random.choice(
                            self.all_aid, replace=False,
                            size=sample_size - len(aid_samples))
                    aid_samples += list(more_ans)
                    
                #aid_samples = np.random.choice(
                #     aidlist, replace=True,
                #    size=int(len(aidlist) * sample_rate))
                #aid_samples = self.q2a[qid]
                #aid_samples.pop(accaid)
                #print(aid_samples)

                rid = self.q2r[qid]
                for x in aid_samples:
                    datalist.append([rid, x, qid])
                    acclist.append(accaid)
        return np.array(datalist), np.array(acclist)

    def __separate_entity(self, entity_seq):
        """
        change a list from "a_1 q_2 r_1 q_2" to three vectors
            a: 1 0 0 0
            q: 0 2 0 2
            r: 0 0 1 0

        args:
            entity_seq  -  the sequence of entities, type=np.array[(str)]

        return:
            three dimensional matrix representing above matrix
        """
        D = {"A": 1, "Q": 2, "R": 0}
        sep = np.zeros(shape=(3, len(entity_seq)))
        for index, item in enumerate(entity_seq):
            split = item.split("_")
            ent_type, ent_id = D[split[0]], int(split[1])
            sep[ent_type][index] = ent_id
        return sep.astype(np.int64)

    def __question_len_emb(self, qid):
        """
        given qid, return the concatenated word vectors

        args:
            qid  -  the qid

        return:
            qvec  -  the vector of the question, numpy.ndarray
            q_len  -  the length of that question
        """
        q_len = 0
        qvecs = [[0.0] * 300 for _ in range(self.PAD_LEN)]
        if qid:
            question = self.question_text[qid]
            question = [x for x in question.strip().split(" ")
                          if x in self.w2vmodel.vocab]
            if question:
                qvecs = self.w2vmodel[question].tolist()
                q_len = len(question)
                if q_len > self.PAD_LEN:
                    qvecs = qvecs[:self.PAD_LEN]
                else:
                    pad_size = self.PAD_LEN - q_len
                    qvecs += [[0.0] * 300 for _ in range(pad_size)]
        return q_len, qvecs

    def __load_word2vec(self):
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

    def __load_question_text(self):
        """
        Load question from dataset, "title" + "content",
            construct the qid2sen dictionary

        Return:
            qid2sen  -  the qid:question dictionary
        """

        qcfile = self.DATA_DIR + "Q_content_nsw.txt"
        qtfile = self.DATA_DIR + "Q_title_nsw.txt"

        qid2sen = {}

        with open(qtfile, "r") as fin_t:
            lines = fin_t.readlines()
            for line in lines:
                id, title = line.split(" ", 1)
                qid2sen[int(id)] = title.strip()

        if self.include_content:
            with open(qcfile, "r") as fin_c:
                lines = fin_c.readlines()
                for line in lines:
                    id, content = line.split(" ", 1)
                    if int(id) in qid2sen:
                        qid2sen[int(id)] += " " + content.strip()
                    else:
                        qid2sen[int(id)] = content.strip()

        return qid2sen

    def __create_uid_index(self):
        """
        Create a uid-index map and index-uid map

        Return:
            uid2ind  -  User id to Index dictionary
            ind2uid  -  Index to User id dictionary
            len(lines)  -  How many users are there in the network
        """
        uid_file = self.DATA_DIR + "QA_ID.txt"
        self.uid2ind[0] = 0
        self.ind2uid[0] = 0
        with open(uid_file, "r") as fin:
            lines = fin.readlines()
            for line in lines:
                ind, uid = line.strip().split(" ")
                ind, uid = int(ind), int(uid)
                self.uid2ind[uid] = ind
                self.ind2uid[ind] = uid
            print("data_loader: user_count", len(lines))
            return len(lines)

    def __load_rqa(self):
        """
        Loading files to create

        Loading Question to Question Raiser ID: self.q2r
                Question to Accepted Answer ID: self.q2acc
                Question to Answer Owner ID: self.q2a (list)
        Return:
            (No return) Just modify the above three dict inplace.
        """
        QR_input = self.DATA_DIR + "Q_R.txt"
        QACC_input = self.DATA_DIR + "Q_ACC.txt"
        QA_input = self.DATA_DIR + "Q_A.txt"

        aid_set = set()

        with open(QR_input, "r") as fin:
            lines = fin.readlines()
            for line in lines:
                Q, R = [int(x) for x in line.strip().split(" ")]
                self.q2r[Q] = R

        with open(QACC_input, "r") as fin:
            lines = fin.readlines()
            for line in lines:
                Q, Acc = [int(x) for x in line.strip().split(" ")]
                self.q2acc[Q] = Acc

        with open(QA_input, "r") as fin:
            lines = fin.readlines()
            for line in lines:
                Q, A = [int(x) for x in line.strip().split(" ")]
                aid_set.add(A)
                if Q not in self.q2a:
                    self.q2a[Q] = [A]
                else:
                    self.q2a[Q].append(A)
        self.all_aid = list(aid_set)

    def __get_question_embeddings(self):
        """
        Quickly load concatenated sentence vectors to from qid

        Return:
            qid2emb  -  the loaded map
        """
        for qid in self.question_text.keys():
            qlen, qvecs = self.__question_len_emb(qid)
            self.qid2emb[qid] = qvecs
            self.qid2len[qid] = qlen
        (zero_len, zero_vecs) = self.__question_len_emb(0)
        self.qid2emb[0] = zero_vecs
        self.qid2len[0] = zero_len 

    def __load_test(self):
        """
        Load test set into memory
        The format of test file is :
            rid, qid, accid

        Return:
            test  -  list of test triples
        """
        test_file = self.DATA_DIR + "test.txt"
        test_set = []
        with open(test_file, "r") as fin:
            lines = fin.readlines()
            for line in lines:
                test_data = [int(x) for x in line.strip().split()]
                rid, qid, accid = test_data[:3]
                aidlist = test_data[3:]
                # rid, qid, accid = [int(x) for x in line.strip().split()]
                test_set.append((rid, qid, accid, aidlist))
        return test_set

    def qid2padded_vec(self, qid_list):
        """
        Convert qid list to a padded sequence.
        The padded length is hard coded into the class

        Args:
            qid_list  -  the list of qid
        Returns:
            padded array  -  the padded array
        """
        qvecs = [self.qid2emb[qid] for qid in qid_list]
        return qvecs
    
    def q2len(self, qid):
        return self.qid2len[qid]

    def qid2vec_length(self, qid_list):
        """
        Convert qid list to list of len
        Args:
            qid_list  -  the list of qid
        Returns:
            len array  -  the list of len
        """
        qlens = [self.qid2len[qid] for qid in qid_list]
        return qlens
    
    def q2emb(self, qid):
        return self.qid2emb[qid]

    def uid2index(self, vec):
        """
        User ID representation to user Index representation

        Args:
            vec  -  the np.array to work with
        Return:
            the transformed numpy array
        """
        def vfind(d, id):
            if id in d:
                return d[id]
            else:
                return random.choice(list(d.values()))
        vfunc = np.vectorize(lambda x: vfind(self.uid2ind, x))
        return vfunc(vec)

    def index2uid(self, vec):
        """
        User Index representation to user ID representation

        Args:
            vec  -  the np.array to work with
        Return:
            the transformed numpy array
        """
        vfunc = np.vectorize(lambda x: self.ind2uid[x])
        return vfunc(vec)

if __name__ == "__main__":
    test = DataLoader(dataset="3dprinting")
    test.__load_word2vec()

