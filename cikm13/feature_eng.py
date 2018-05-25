"""
Feature engineering-based Question Routing

Author:
    Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>

Implementing CIKM'13 paper:
    Zongcheng Ji and Bin Wang
    Learning to Rank for Question Routing in Community Question Answering
"""

import os, sys

from collections import Counter
try:
    import json as ujson
except:
    import json

# Global Variables
DATA_DIR = os.getcwd() + "/data/"
PARSED_DIR = DATA_DIR + "parsed/"
CUR_DIR = os.getcwd() + "/cikm13/"


def question_title_len(dataset):
    infile = PARSED_DIR + "{}/Q_title.txt".format(dataset)
    outfile = CUR_DIR + "question.title.length"
    with open(infile, "r") as fin,\
        open(outfile, "w") as fout:
        lines = fin.readlines()
        lines = [line.strip().split(" ") for line in lines]
        for x in lines:
            print("{} {}".format(x[0], len(x) - 1), file=fout)

def user_specific(dataset):
    infile = DATA_DIR + "Record_Train.json"
    outfile = CUR_DIR + "user.specific"

    user_answers = Counter()
    user_best_answers = Counter()
    user_asks = Counter()
    all_users = set()
    with open(infile, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            data = json.loads(line)
            rid = int(data['QuestionOwnerId'])
            accaid = int(data['AcceptedAnswererId'])
            aidlist = [int(x) for x in data['AnswererIdList']]

            all_users.add(rid)
            all_users.update(aidlist)

            user_answers.update(aidlist)
            user_best_answers.update([accaid])
            user_asks.update([rid])







