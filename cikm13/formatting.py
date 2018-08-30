import os
import numpy as np
import pandas as pd
import json


def question_features(dataset):
    dir = os.getcwd() + "data/" + dataset + "/"

    # ============ LDA Feature ====================
    qid_lda = {}
    with open(dir + "qid.prob.lda", "r") as fin:
        for line in fin.readlines():
            qid, feature = line.strip().split(" ")
            qid_lda[int(qid)] = float(feature)
        qid_lda[-1] = np.mean(list(qid_lda.values()))  # default value

    # ============ LM ANS Feature ====================
    qid_lm_ans = {}
    with open(dir + "qid.prob.lm.ans", "r") as fin:
        for line in fin.readlines():
            qid, feature = line.strip().split(" ")
            qid_lm_ans[int(qid)] = float(feature)
        qid_lm_ans[-1] = np.mean(list(qid_lm_ans.values()))  # default values

    # ============ LM ASK_ANS Feature ====================
    qid_lm_ask_ans = {}
    with open(dir + "qid.prob.lm.ask_ans", "r") as fin:
        for line in fin.readlines():
            qid, feature = line.strip().split(" ")
            qid_lm_ask_ans[int(qid)] = float(feature)
        qid_lm_ask_ans[-1] = np.mean(list(qid_lm_ask_ans.values()))  # default values

    # ============ Question Length Feature ====================
    qid_len = {}
    with open(dir + "question.title.length", "r") as fin:
        for line in fin.readlines():
            qid, length = line.strip().split(" ")
            qid_len[int(qid)] = int(length)
        qid_len[-1] = np.mean(list(qid_len.values()),
                              dtype=np.int32)  # default values

    user_feature = {}
    df = pd.read_csv("test", delim_whitespace=True, header=None)
    defaults = [df[x].mean() for x in range(1, 4)]
    with open(dir + "user.specific", "r") as fin:
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
                             + create_substring(user_features=user_feature, uid=uid)
                    out_buffer.append(term_a)
            except KeyError:
                pass
    out_file = dir + "train.dat"
    with open(out_file, "w") as fout:
        for out_line in out_buffer:
            fout.write(out_line)


def create_substring( user_features, uid):
    features = user_features.get(uid, user_features[-1])
    features_str_list = ["{}:{}".format(i+5, features[i]) for i in range(len(features))]
    return " ".join(features_str_list)

