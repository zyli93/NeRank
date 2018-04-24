"""
    Data Loader

    the class of loading data

    author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>

"""

import numpy as np
import os, sys

import gensim
import random

data_index = 0
test_index = 0

class DataLoader():
    def __init__(self, dataset, include_content, mp_coverage, mp_length):
        print("Initializing data_loader ...")
        self.PAD_LEN = 24
        self.dataset = dataset
        self.include_content = include_content
        self.mpfile = os.getcwd() \
                + "/metapath/"+ self.dataset + "_" + str(mp_coverage) \
                + "_" + str(mp_length) + ".txt" 
        self.datadir = os.getcwd() + "/data/parsed/" + self.dataset + "/"

        print("\tloading dataset ...")
        data = self.__read_data()
        self.train_data = data
        print("\tcounting dataset ...")
        self.count = self.__count_dataset(data)
        print("\tinitializing sample table ...")
        self.sample_table = self.__init_sample_table()
        print("\tloading word2vec model ...")
        self.w2vmodel = self.__load_word2vec()  # **time consuming!**
        print("\tloading questions ...")
        self.qid2sen = self.__load_questions()

        print("\tcreating user-index mapping ...")
        self.uid2ind, self.ind2uid = {}, {}
        self.user_count = self.__create_uid_index()

        self.q2r, self.q2acc, self.q2a = {}, {}, {}
        self.all_aid = []
        print("\tloading rqa ...")
        self.__load_rqa()

        print("\tcreating qid embeddings map ...")
        self.qid2emb, self.qid2len = {}, {}
        self.__qid2embedding()

        print("\tloading test sets ...")
        self.testset = self.__load_test()
        print("\tloading test set q-a pairs ...")
        self.testqa = self.__load_test_qa()

        self.process = True

        print("Done - Data Loader!")

    def __read_data(self):
        """
        read metapath dataset,
            load the dataset into data

        return:
            data  -  the metapath dataset
        """
        with open(self.mpfile, "r") as fin:
            lines = fin.readlines()
            data = [line.strip().split(" ") for line in lines]
            return data

    def __count_dataset(self, data):
        """
        read dataset and count the frequency

        args:
            data  -  the list of meta-pathes.
        returns:
            count  - the sorted list of
        """
        count_dict = {}
        for path in data:
            for entity in path:
                count_dict[entity] = count_dict.get(entity, 0) + 1

        # keys: entity id, values: count
        count = list(zip(count_dict.keys(), count_dict.values()))
        count.sort(key=lambda x:x[1], reverse=True)
        return count

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

    def generate_batch(self, window_size, batch_size, neg_ratio):
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
        data = self.train_data
        global data_index
        pairs_list = []

        batch = batch_size if batch_size + data_index < len(data) \
            else len(data) - data_index

        for i in range(batch):
            pairs = self.__slide_through(data_index, window_size)
            if data_index + 1 < len(data):
                data_index += 1
            else:  # meet the end of the dataset
                self.process = False
                data_index = 0
                break
            pairs_list += pairs

        try:
            u, v = zip(*pairs_list)
        except:
            print(pairs_list)
            pass
        upos = self.__separate_entity(u)
        vpos = self.__separate_entity(v)

        npairs_in_batch = len(pairs_list)
        neg_samples = np.random.choice(
            self.sample_table,
            # first get a long neg sample list
            # then after separating entity, reshape to 3xlxh
            size=int(npairs_in_batch * neg_ratio))

        # why:
        #   instead of returning a mat, here it return a long np.array.
        #   in the model, it reshape.
        # npos = self.__separate_entity(neg_samples).reshape(
        #     3,  # raq for 3 sub-matrices
        #     npairs_in_batch * 2 * window_size,
        #     int(npairs_in_batch * neg_ratio))
        npos = self.__separate_entity(neg_samples)

        aqr, accqr = self.get_acc_ind(upos, vpos)
        return upos, vpos, npos, npairs_in_batch, aqr, accqr

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

    def __slide_through(self, ind, window_size):
        """
        sliding through one span, generate all pairs in it

        args:
            ind  -  the index of the label entity
            window_size  -  the window_size of context
        return:
            the pair list
        """
        meta_path = self.train_data[ind]
        pairs = []
        for pos, token in enumerate(meta_path):
            lcontext, rcontext = [], []
            lcontext = meta_path[pos - window_size: pos] \
                if pos - window_size >= 0 \
                else meta_path[:pos]

            if pos + 1 < len(meta_path):
                rcontext = meta_path[pos + 1 : pos + window_size] \
                    if pos + window_size < len(meta_path) \
                    else meta_path[pos + 1 :]

            context_pairs = [[token, context]
                             for context in lcontext + rcontext]
            pairs += context_pairs
        return pairs

    def __qid_to_concatenate_emb(self, qid):
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
            question = self.qid2sen[qid]
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

    def __load_questions(self):
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
                qid2sen[int(id)] = title.strip()

        if self.include_content:
            with open(qcfile, "r") as fin_c:
                lines = fin_c.readlines()
                for line in lines:
                    id, content = line.split(" ", 1)
                    qid2sen[int(id)] += " " + content.strip()

        return qid2sen

    def __create_uid_index(self):
        """
        Create a uid-index map and index-uid map

        Return:
            uid2ind  -  User id to Index dictionary
            ind2uid  -  Index to User id dictionary
            len(lines)  -  How many users are there in the network
        """
        uid_file = self.datadir + "part_users.txt"
        self.uid2ind[0] = 0
        self.ind2uid[0] = 0
        with open(uid_file, "r") as fin:
            lines = fin.readlines()
            for ind, line in enumerate(lines):
                uid = int(line.strip())
                self.uid2ind[uid] = ind
                self.ind2uid[ind] = uid
            return len(lines)

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

    def __load_rqa(self):
        """
        Loading files to create

        Loading Question to Question Raiser ID: self.q2r
                Question to Accepted Answer ID: self.q2acc
                Question to Answer Owner ID: self.q2a (list)
        Return:
            (No return) Just modify the above three dict inplace.
        """
        QR_input = self.datadir + "Q_R.txt"
        QACC_input = self.datadir + "Q_ACC_A.txt"
        QA_input = self.datadir + "Q_A.txt"

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

    def get_acc_ind(self, upos, vpos):
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
            if upos[1][i] and vpos[2][i]:
                aid = upos[1][i]
                qid = vpos[2][i]
                # Check this AQ relationship make sence
                if aid in self.q2a[qid]:
                    accaid = self.q2acc[qid]
                    rid = self.q2r[qid]
                    datalist.append([rid, aid, qid])
                    acclist.append(accaid)
        return np.array(datalist), np.array(acclist)

    def __qid2embedding(self):
        """
        Quickly load concatenated sentence vectors to from qid

        Return:
            qid2emb  -  the loaded map
        """
        for qid in self.qid2sen.keys():
            qlen, qvecs = self.__qid_to_concatenate_emb(qid)
            self.qid2emb[qid] = qvecs
            self.qid2len[qid] = qlen
        (zero_len, zero_vecs) = self.__qid_to_concatenate_emb(0)
        self.qid2emb[0] = zero_vecs
        self.qid2len[0] = zero_len 

    def q2emb(self, qid):
        """
        Getter func of qid2emb
        Args:
            qid  -  Hello
        """
        return self.qid2emb[qid]
    
    def q2len(self, qid):
        """
        Getter func of qid2len
        Args:
            qid  - Hello
        """
        return self.qid2len[qid]

    def __load_test(self):
        """
        Load test set into memory
        The format of test file is :
            rid, qid, accid

        Return:
            test  -  list of test triples
        """
        test_file = self.datadir + "test.txt"
        test_set = []
        with open(test_file, "r") as fin:
            lines = fin.readlines()
            for line in lines:
                test_data = [int(x) for x in line.strip().split()]
                rid, qid, accid = test_data[:3]
                aids = test_data[3:]
                # rid, qid, accid = [int(x) for x in line.strip().split()]
                test_set.append((rid, qid, accid, aids))
        return test_set

    def build_test_batch(self, test_prop, test_neg_ratio):
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
            # inds = np.arange(total)
            # batch_inds = np.random.choice(inds, batch_size, replace=False)
            # batch = self.testset[batch_inds]
        else:
            batch = self.testset

        # test_batch = []
        # for test_sample in batch:
        #     rid, qid, accaid, aids = test_sample
        #     alist = self.testqa[qid]
        #     Sample some negative
        #     neg_alist = np.random.choice(self.all_aid,
        #                                  test_neg_ratio * len(alist),
            #                              replace=False)
            # trank_a = []
            # for aid in alist + neg_alist.tolist():
            #     trank_a.append(aid)
            # test_batch.append([rid, qid, accaid, trank_a])
        return batch

    def perform_metric(self, aid_list, score_list, accid, k):
        """
        Performance metric evaluation

        Args:
            aid_list  -  the list of aid in this batch
            score_list  -  the list of score of ranking
            accid  -  the ground truth
            k  -  precision at K
        """
        if len(aid_list) != len(score_list):
            print("aid_list and score_list not equal length.",
                  file=sys.stderr)
            sys.exit()
        id_score_pair = list(zip(aid_list, score_list))
        id_score_pair.sort(key=lambda x: x[1], reverse=True)
        for ind, (aid, score) in enumerate(id_score_pair):
            if aid == accid:
                return 1/(ind+1), int(ind < k)

    def qid2vec_padded(self, qid_list):
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

    def qid2vec_len(self, qid_list):
        """
        Convert qid list to list of len
        Args:
            qid_list  -  the list of qid
        Returns:
            len array  -  the list of len
        """
        qlens = [self.qid2len[qid] for qid in qid_list]
        return qlens
    
    def __load_test_qa(self):
        """
        Load answers list of questions in test set

        Args:
        Returns:
            test_qa  -  the qa set of the test dataset
        """
        test_qa = {}
        with open(self.datadir + "test_q_alist.txt", "r") as fin:
            lines = fin.readlines()
            for line in lines:
                pair = [int(x) for x in line.strip().split(" ")]
                qid, aid = pair[0], pair[1]
                if qid not in test_qa:
                    test_qa[qid] = [aid]
                else:
                    test_qa[qid].append(aid)
        return test_qa

if __name__ == "__main__":
    test = DataLoader(dataset="3dprinting")
    test.__load_word2vec()

