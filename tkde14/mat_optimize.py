"""
Matrix Completion Question Routing

Author:
    Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>

Implementing TKDE'14 paper:
    Zhou Zhao, et al.
    Expert Finding for Question Answering via Graph Regularized Matrix Completion
"""

import cPickle

from scipy.sparse import coo_matrix, save_npz

import os, sys
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

try:
    import ujson as json
except:
    import json

# Global Variables
DATA_DIR = os.getcwd() + "/data/"
PARSED_DIR = DATA_DIR + "parsed/"
CUR_DIR = os.getcwd() + "/tkde14/"
CUR_PARSED_DIR = CUR_DIR + "parsed/"

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
    infile = PARSED_DIR + "{}/Q_content_nsw.txt".format(dataset)
    mat_Q_file = CUR_DIR + "mat.Q"
    question_index_file = CUR_DIR + "question.to.index"

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
    mat_Y_file = CUR_DIR + "mat.Y"

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

