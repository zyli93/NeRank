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

from preprocessing import clean_str2, remove_stopwords

nltk.data.path.append("/workspace/nltk_data")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Global Variables
DATA_DIR = os.getcwd() + "/data/"
PARSED_DIR = DATA_DIR + "parsed/"
CUR_DIR = os.getcwd() + "/cikm13/"
CUR_PARSED_DIR = CUR_DIR + "/parsed/"


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
    infile = DATA_DIR + "{}/Record_Train.json".format(dataset)
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

    """
    Extracting four user-specific features 
        1. % of best answer
        2. # of best answers
        3. # of answers
        4. # of asks
    """
    all_users = sorted(list(all_users))
    with open(outfile, "w") as fout:
        for user_id in all_users:
            n_best = user_best_answers[user_id]
            n_answer = user_answers[user_id]
            n_ask = user_asks[user_id]
            percentage = n_best / n_answer if n_answer else 0
            print("{:d} {:.6f} {:d} {:d} {:d}".format(
                user_id, percentage, n_best, n_answer, n_ask), file=fout)


def question_user(dataset):
    infile_a = DATA_DIR + "{}/Posts_A.json".format(dataset)
    infile_q = DATA_DIR + "{}/Posts_Q.json".format(dataset)

    outfile = CUR_PARSED_DIR + "A.answer.content"

    generate_outfile = True
    if not os.path.exists(CUR_PARSED_DIR):
        os.mkdir(CUR_PARSED_DIR)

    if os.path.exists(outfile):
        print("Answer content file exists. Skipping generation")
        generate_outfile = False

    if generate_outfile:
        user_content = {}
        sw_set = set(stopwords.words('english'))

        with open(infile_a, "r") as fin:
            lines = fin.readlines()
            for line in lines:
                data = json.loads(line)
                aid = data.get('OwnerId', None)
                if not aid:
                    continue
                body = remove_stopwords(
                        clean_str2(data['Body']), sw_set)
                if aid not in user_content:
                    user_content[aid] = [body]
                else:
                    user_content[aid].append(body)

        with open(infile_q, "r") as fin:
            lines = fin.readlines()
            for line in lines:
                data = json.loads(line)
                rid = data.get('OwnerId', None)
                if not rid:
                    continue
                body = remove_stopwords(
                        clean_str2(data['Body']), sw_set)
                if rid not in user_content:
                    user_content[rid] = [body]
                else:
                    user_content[rid].append(body)

        with open(outfile, "w") as fout:
            for uid in user_content.keys():
                text = " ".join(user_content[uid])
                print("{} {}".format(uid, text), file=fout)

    # Later computation















