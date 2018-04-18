"""
    Data Loader

    the class of loading data

    author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>

"""

import numpy as np
import os, sys

import gensim

data_index = 0

class DataLoader():
    def __init__(self, dataset, include_content):
        print("initializing data_loader ...")
        self.dataset = dataset
        self.include_content = include_content
        self.mpfile = os.getcwd() + "/metapath/"+ self.dataset +".txt"
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
        print("\tloading rqa ...")
        self.__load_rqa()

        print("\tcreating qid embeddings map")
        self.qid2emb = self.__qid2embedding()

        self.process = True

        print("done!")

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
        table_size = 1e6 # todo: what is this???
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

        u, v = zip(*pairs_list)
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
        """
        if qid:
            question = self.qid2sen[qid]
            question = [x for x in question.strip().split(" ")
                          if x in self.w2vmodel.vocab]
            if not question:
                qvecs = np.random.random(300).reshape((1, 300))
            else:
                qvecs = self.w2vmodel[question]
        else:
            qvecs = np.zeros((1, 300))
        return qvecs

    def qtc(self, qid):
        return self.__qid_to_concatenate_emb(qid)

    def qid2vec(self, vec):
        """
        Vector qid to a vector of concatenated word embeddings

        Args:
            vec  -  the vector of qids to deal with

        Return:
            the list of concatenated word embeddings.
            Each element should be variant in sizes.
        """
        vfunc = np.vectorize(lambda x: self.__qid_to_concatenate_emb(x))
        return vfunc(vec)

    def sen2vecs(self, sentence):
        """
        ** [NOT in USE] **
        Convert sentence to concatenate of word-vectors

        Arg:
            sentence  -  str, the sentence to convert

        Return:
            (np.array)  -  the array contains the word vectors
        """
        sentence = sentence.split(" ")
        vectors = [self.w2vmodel[w] for w in sentence]
        return len(sentence), np.array(vectors)

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

    def single_uid2index(self, uid):
        """ NOT IN USE"""
        return self.uid2ind[uid]

    def single_index2uid(self, index):
        """NOT IN USE"""
        return self.ind2uid[index]

    def uid2index(self, vec):
        """
        User ID representation to user Index representation

        Args:
            vec  -  the np.array to work with
        Return:
            the transformed numpy array
        """
        vfunc = np.vectorize(lambda x: self.uid2ind[x])
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
                if Q not in self.q2a:
                    self.q2a[Q] = [A]
                else:
                    self.q2a[Q].append(A)

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
        qid2emb  = {}
        for qid in self.q2r.keys():
            qid2emb[qid] = self.__qid_to_concatenate_emb(qid)
        return qid2emb

    def q2emb(self, qid):
        """
        Getter func of qid2emb
        Args:
            qid  -  Hello
        """
        return self.qid2emb[qid]

if __name__ == "__main__":
    test = DataLoader(dataset="3dprinting")
    test.__load_word2vec()

