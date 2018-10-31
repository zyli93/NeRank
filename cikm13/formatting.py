import os
import numpy as np
import pandas as pd
import json

import sys


qid_lda = {}
qid_lm_ans = {}
qid_lm_ask_ans = {}
qid_len = {}
user_feature = {}

def question_features(dataset):
    DIR = os.getcwd() + "/data/" + dataset + "/"

    # ============ LDA Feature ====================
    with open(DIR + "qid.prob.lda", "r") as fin:
        for line in fin.readlines():
            qid, feature = line.strip().split(" ")
            qid_lda[int(qid)] = float(feature)
        qid_lda[-1] = np.mean(list(qid_lda.values()))  # default value

    # ============ LM ANS Feature ====================
    with open(DIR + "qid.prob.lm.ans", "r") as fin:
        for line in fin.readlines():
            qid, feature = line.strip().split(" ")
            qid_lm_ans[int(qid)] = float(feature)
        qid_lm_ans[-1] = np.mean(list(qid_lm_ans.values()))  # default values

    # ============ LM ASK_ANS Feature ====================
    with open(DIR + "qid.prob.lm.ask_ans", "r") as fin:
        for line in fin.readlines():
            qid, feature = line.strip().split(" ")
            qid_lm_ask_ans[int(qid)] = float(feature)
        qid_lm_ask_ans[-1] = np.mean(list(qid_lm_ask_ans.values()))  # default values

    # ============ Question Length Feature ====================
    with open(DIR + "question.title.length", "r") as fin:
        for line in fin.readlines():
            qid, length = line.strip().split(" ")
            qid_len[int(qid)] = int(length)
        qid_len[-1] = np.mean(list(qid_len.values()),
                              dtype=np.int32)  # default values

    df = pd.read_csv(DIR + "user.specific", delim_whitespace=True, header=None)
    defaults = [df[x].mean() for x in range(1, 4)]
    with open(DIR + "user.specific", "r") as fin:
        for line in fin.readlines():
            elements = line.strip().split(" ")
            user_feature[int(elements[0])] = [float(x) for x in elements[1:]]
        user_feature[-1] = defaults

    in_file = os.getcwd() + "/../data/{}/Record_Train.json".format(dataset)
    out_buffer = []
    with open(in_file, "r") as fin:
        for query_index, line in enumerate(fin.readlines()):
            data = json.loads(line)
            try:
                qid = int(data["QuestionId"])
                q_feature = [
                    qid_lda.get(qid, qid_lda[-1]),
                    qid_lm_ans.get(qid, qid_lm_ans[-1]),
                    qid_lm_ask_ans.get(qid, qid_lm_ask_ans[-1]),
                    qid_len.get(qid, qid_len[-1])
                ]
                q_feature_string = " ".join(["{}:{}".format(index + 1, value)
                                             for index, value in enumerate(q_feature)])
                acc_aid = int(data['AcceptedAnswererId'])
                aid_list = [int(x) for x in data['AnswererIdList']]

                # Create Acc terms
                term_acc = "2 qid:{} ".format(query_index) \
                           + q_feature_string + " " \
                           + create_substring(user_features=user_feature, uid=acc_aid)
                out_buffer.append(term_acc)

                # Create Answer terms
                for uid in aid_list:
                    term_a = "1 qid:{} ".format(query_index) \
                             + q_feature_string + " " \
                             + create_substring(user_features=user_feature, uid=uid)
                    out_buffer.append(term_a)
            except KeyError:
                pass
    out_file = DIR + "train.dat"
    with open(out_file, "w") as fout:
        for out_line in out_buffer:
            fout.write(out_line + "\n")


def create_substring(user_features, uid):
    features = user_features.get(uid, user_features[-1])
    features_str_list = ["{}:{}".format(i+5, features[i]) for i in range(len(features))]
    return " ".join(features_str_list)


def create_test(dataset):
    DIR = os.getcwd() + "/data/" + dataset + "/"

    in_file = os.getcwd() + "/../data/parsed/{}/test.txt".format(dataset)
    output_buffer = []
    with open(in_file, "r") as fin:
        for query_index, line in enumerate(fin.readlines()):
            line_split = line.strip().split(" ")
            qid = int(line_split[1])
            accaid = int(line_split[2])
            uids = [int(x) for x in line_split[3:]]
            uids.pop(uids.index(accaid))
            q_feature = [
                qid_lda.get(qid, qid_lda[-1]),
                qid_lm_ans.get(qid, qid_lm_ans[-1]),
                qid_lm_ask_ans.get(qid, qid_lm_ask_ans[-1]),
                qid_len.get(qid, qid_len[-1])
            ]
            q_feature_string = " ".join(["{}:{}".format(index + 1, value)
                                         for index, value in enumerate(q_feature)])

            output_buffer.append("1 qid:{} ".format(query_index) + q_feature_string + " " +
                                 create_substring(user_features=user_feature, uid=accaid))
            for uid in uids:
                output_buffer.append("1 qid:{} ".format(query_index) + q_feature_string + " " +
                                     create_substring(user_features=user_feature, uid=uid))

    with open(DIR + "test.dat", "w") as fout:
        for x in output_buffer:
            fout.write(x + "\n")


def evaluate(dataset, count):
    in_file = os.getcwd() + "/predictions/" + dataset[:3]
    MRR,  hit_K, prec_1 = 0, 0, 0
    with open( in_file, "r") as fin:
        lines = fin.readlines()
        lines = [float(x.strip()) for x in lines]
        queries = [lines[i: i+count] for i in range(0, len(lines), count)]
        count = 0
        for qry in queries:
            gt = qry[0]
            qry.sort()
            rank = qry.index(gt) + 1
            MRR += 1/rank
            hit_K += 1 if rank < 5 else 0
            prec_1 += 1 if rank == 1 else 0
            count += 1
        print("MRR: {:.6f}, HaK {:.6f}, Pa1: {:.6f}".format(
            MRR / count , hit_K / count, prec_1 / count))



if __name__ == "__main__":
    if len(sys.argv) < 2 + 1:
        print("Invalid input")
        print("\tparameter list: [dataset] [mode] [Count]")
        sys.exit(1)
    ds = sys.argv[1]
    mode = int(sys.argv[2])
    count = int(sys.argv[3])

    if mode == 1:
        print("Doing mode 1-1")
        question_features(ds)
        print("Doing mode 1-2")
        create_test(ds)
    elif mode == 2:
        print("Doing mode 2-1")
        print(evaluate(ds, count))


