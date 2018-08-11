"""
Matrix Completion Question Routing

Author:
    Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>

Implementing TKDE'14 paper:
    Zhou Zhao, et al.
    Expert Finding for Question Answering via Graph Regularized Matrix Completion
"""

from scipy.sparse import coo_matrix, save_npz
from collections import Counter

import os, sys
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import itertools

try:
    import ujson as json
except:
    import json

# Global Variables
DATA_DIR = os.getcwd() + "/data/"
PARSED_DIR = DATA_DIR + "parsed/"
CUR_DIR = os.getcwd() + "/tkde14/data/"

"""
Three files to generate:
    m: # of questions, 
    n: # of users,
    d: # of words in vocabulary
    
    1 - question.to.index: question ID to matrix index
    2 - mat.Q (mat.Q.npz): question representation w.r.t words (d x m)
    3 - mat.Y (mat.Y): 
    
"""


def build_matrix_Q(dataset):
    infile = PARSED_DIR + "{}/Q_title_nsw.txt".format(dataset)
    mat_Q_file = CUR_DIR + "{}/mat.Q".format(dataset)
    question_index_file = CUR_DIR + "{}/question.to.index".format(dataset)

    count_vectorizer = CountVectorizer(
        analyzer='word', stop_words='english', max_df=0.9, min_df=2)

    with open(infile, "r") as fin,\
        open(question_index_file, "w") as fout:
        lines = fin.readlines()
        lines = [line.split(" ", 1) for line in lines]
        question_id_list = [entry[0] for entry in lines]
        txt = [entry[1] for entry in lines]

        print("\tFit-Transforming texts...", end= " ")
        mat_Q = count_vectorizer.fit_transform(txt).toarray()
        mat_Q = np.transpose(mat_Q)  # to make it d x m

        for index, qid in enumerate(question_id_list):
            print("{} {:d}".format(qid, index), file=fout)

        sparse_mat_Q = coo_matrix(mat_Q)
        save_npz(mat_Q_file, sparse_mat_Q)

        print("Done!")


def build_matrix_Y(dataset):
    infile = DATA_DIR + "{}/Posts_A.json".format(dataset)
    mat_Y_file = CUR_DIR + "{}/mat.Y".format(dataset)

    """
    Format of matrix Y:
        user_id question_id votes
    """

    with open(infile, "r") as fin,\
        open(mat_Y_file, "w") as fout:
        lines = fin.readlines()
        for line in lines:
            data = json.loads(line)
            uid = data.get('OwnerUserId', None)
            qid = data.get('ParentId', None)
            score = data.get('Score', None)
            if uid and qid and score:
                print("{} {} {}".format(uid, qid, score), file=fout)


def build_matrix_L(dataset):
    """
    Build matrix L,
        L = D - W
        D_ii = \sum_j W_ij
        W_ij = Directed + Undirected

    Args:
        dataset - the dataset

    Return:
        write the sparse matrix L to file
    """
    infile = DATA_DIR + "{}/Record_Train.json".format(dataset)
    outfile = CUR_DIR + "{}/mat.L".format(dataset)

    # Compute W

    ask_ans_links = set()  # Raiser & Answerer, modeling the friend relationship
    user_coanswer = Counter()
    user_answer = Counter()
    with open(infile, "r") as fin:
        lines = fin.readlines()

        for line in lines:
            data = json.loads(line)
            rid = int(data['QuestionOwnerId'])
            aidlist = [int(x) for x in data['AnswererIdList']]
            ask_ans_links.update(
                [(rid, aid) if rid <= aid else (aid, rid) for aid in aidlist])

            user_answer.update(aidlist)  # Record # of questions a user answered
            coanswers = list(itertools.combinations(aidlist, 2))  # Co-answered
            user_coanswer.update(coanswers)

    dict_W = dict((pair, 1) for pair in ask_ans_links)
    max_id = np.amax(list(ask_ans_links)) + 1

    for t in user_coanswer.keys():
        if t in ask_ans_links:
            continue
        aid1, aid2 = t[0], t[1]
        dict_W[t] = user_coanswer[t] \
                    / (user_answer[aid1] + user_answer[aid2] - user_coanswer[t])

    dim1 = [x[0][0] for x in dict_W.items()]
    dim2 = [x[0][1] for x in dict_W.items()]
    row = np.array(dim1 + dim2)
    col = np.array(dim2 + dim1)
    val = np.array([x[1] for x in dict_W.items()] * 2)
    mat_W = coo_matrix((val, (row, col)), shape=(max_id, max_id))
    mat_D_value = mat_W.sum(axis=1).squeeze().tolist()[0]
    mat_W = - mat_W
    mat_W.setdiag(mat_D_value)
    save_npz(outfile, mat_W)


if __name__ == "__main__":
    if len(sys.argv) < 1 + 1:
        print("Usage:\n\tpython {} [dataset]".format(sys.argv[0]))
        sys.exit(1)

    dataset = sys.argv[1]

    RESULT_DIR = CUR_DIR + dataset
    if not os.path.exists(RESULT_DIR):
        os.mkdir(CUR_DIR)
        os.mkdir(RESULT_DIR)

    print("Building Matrix L...")
    build_matrix_L(dataset)

    print("Building Matrix Q...")
    build_matrix_Q(dataset)

    print("Building Matrix Y...")
    build_matrix_Y(dataset)

    print("Output files are stored in {}".format(RESULT_DIR))
    print("Done!")





