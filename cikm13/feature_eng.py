"""
Feature engineering-based Question Routing

Author:
    Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>

Implementing CIKM'13 paper:
    Zongcheng Ji and Bin Wang
    Learning to Rank for Question Routing in Community Question Answering
"""

import os, sys
import _pickle as pickle
import math
from collections import Counter
import numpy as np

import nltk
nltk.data.path.append("/workspace/nltk_data")
from nltk.corpus import stopwords

from gensim.corpora import Dictionary
from gensim.models import ldamodel

try:
    import ujson as json
except:
    import json

from preprocessing import clean_str2, remove_stopwords



# Global Variables
DATA_DIR = os.getcwd() + "/data/"
PARSED_DIR = DATA_DIR + "parsed/"
CUR_DIR = os.getcwd() + "/cikm13/data/"


def question_title_len(dataset):
    infile = PARSED_DIR + "{}/Q_title.txt".format(dataset)
    outfile = CUR_DIR + "{}/question.title.length".format(dataset)
    with open(infile, "r") as fin,\
        open(outfile, "w") as fout:
        lines = fin.readlines()
        lines = [line.strip().split(" ") for line in lines]
        for x in lines:
            print("{} {}".format(x[0], len(x) - 1), file=fout)


def user_specific(dataset):
    infile = DATA_DIR + "{}/Record_Train.json".format(dataset)
    outfile = CUR_DIR + "{}/user.specific".format(dataset)

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


def question_user_content(dataset):
    """
    Generating user - answer and
               user - question+answer contents

    Args:
        dataset  -  the dataset
    """

    def record_append(d, user_id, qid, text):
        user_id = int(user_id)
        qid = int(qid)
        if user_id not in d:
            d[user_id] = [(qid, text)]
        else:
            d[user_id].append((qid, text))

    qa_map_file = DATA_DIR + "{}/Record_All.json".format(dataset)
    infile_q = DATA_DIR + "{}/Posts_Q.json".format(dataset)

    user_ans_file = CUR_DIR + \
                    "{}/user.ans.question".format(dataset)
    user_ask_ans_file = CUR_DIR + \
                      "{}/user.ask_ans.question".format(dataset)

    if not os.path.exists(CUR_DIR):
        os.mkdir(CUR_DIR)

    if os.path.exists(user_ask_ans_file)\
            and os.path.exists(user_ans_file):
        print("\n\t\tAnswer content file exists. Skipping generation",
              end=" ")
        return

    user_ansd = {}  # User answered
    user_ansd_askd = {}  # User answered and asked
    question_content = {}  # The content of the question
    sw_set = set(stopwords.words('english'))

    with open(infile_q, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            data = json.loads(line)
            qid = data.get('Id', None)
            qcontent = data.get('Body', None)
            rid = data.get('OwnerUserId', None)
            if not (qid and qcontent and rid):
                continue

            qcontent = remove_stopwords(clean_str2(qcontent), sw_set)
            record_append(user_ansd_askd, rid, qid, qcontent)
            question_content[qid] = qcontent

    with open(qa_map_file, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            data = json.loads(line)
            aid_list = data['AnswererIdList']
            qid = data['QuestionId']
            for aid in aid_list:
                record_append(user_ansd_askd, aid, qid, question_content[qid])
                record_append(user_ansd, aid, qid, question_content[qid])

    with open(user_ans_file, "wb") as fout:
        pickle.dump(user_ansd, fout)

    with open(user_ask_ans_file, "wb") as fout:
        pickle.dump(user_ansd_askd, fout)


def question_user_features(dataset, mode):
    if mode not in ['lm.ans', 'lm.ask_ans', 'lda']:
        print("Incorrect mode provided!")
        sys.exit(1)

    mode_file = "ans" if mode in ['lm.ans', 'lda'] else "ask_ans"
    infile = CUR_DIR + "{}/user.{}.question".format(dataset, mode_file)
    outfile = CUR_DIR + "{}/qid.prob.{}".format(dataset, mode)

    with open(infile, "rb") as fin:
        user_records = pickle.load(fin)
    if not user_records:
        print("User records of {} is empty".format(mode))
        sys.exit(1)

    uid_list = user_records.keys()
    prob_func = get_prob_LDA if mode == "lda" else get_prob_LM
    with open(outfile, "w") as fout:
        for uid in uid_list:
            records = user_records[uid]
            probs = prob_func(records)
            for qid, prob in probs:
                try:
                    print("{:d} {:.6f}".format(qid, prob), file=fout)
                except:
                    print(uid)
                    print(" ", qid)
                    print(type(qid))
                    print(prob_func)
                    print(mode)
                    sys.exit()


def get_prob_LM(records):
    """
    Get the probability of each record.

    Args:
        records - (qid, text) tuples

    Returns:
        probs - probability in terms of (qid, prob)
    """
    all_text = " ".join([x[1] for x in records])
    one_grams = all_text.split(" ")
    og_count = Counter(one_grams)
    og_total = len(one_grams)

    two_grams = list(zip(one_grams, one_grams[1:]))
    tg_count = Counter(two_grams)
    tg_start_count = Counter([x[0] for x in two_grams])

    probs = []

    for qid, text in records:
        text = text.split(" ")
        init_token = text[0]
        trans_tokens = list(zip(text, text[1:]))

        log_prob = math.log(og_count[init_token] / og_total)
        for trans in trans_tokens:
            margin = tg_start_count[trans[0]]
            joint  = tg_count[trans]
            log_prob += math.log(joint / margin)
        probs.append((qid, log_prob))

    return probs


def get_prob_LDA(records):
    """
    Args:
        records - (qid, text)
    """
    num_topics = 10
    topn_words = 1000

    probs = []

    qid_list = [int(x[0]) for x in records]
    texts = [x[1].split(" ") for x in records]

    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    np.random.seed(1)

    model = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=num_topics)

    topic_word_probs = [
        dict(model.get_topic_terms(topic_id, topn=topn_words))
        for topic_id in range(num_topics)]

    for index, text in enumerate(texts):
        text_prob = 0.0
        text_bow = model.id2word.doc2bow(text)
        doc_topics, word_topics, phi_values = \
            model.get_document_topics(text_bow, per_word_topics=True)

        # tprob is the hash table of [topic ID: topic probability]
        tprob = dict(doc_topics)

        # wtopic is the hash table of [word ID: list of topics it is in]
        wtopic = dict(word_topics)

        for word_id, count in text_bow:
            word_prob = [(tprob[topic_id] if topic_id in tprob else 0) *
                         (topic_word_probs[topic_id][word_id] if word_id in topic_word_probs[topic_id] else 0)
                        for topic_id in wtopic[word_id]]
            word_prob = sum(word_prob)
            # print(word_prob)
            text_prob += count * math.log(word_prob + 0.0001)
        probs.append((qid_list[index], text_prob))

    return probs


if __name__ == "__main__":
    if len(sys.argv) < 1 + 1:
        print("Usage:\n\tpython {} [dataset]".format(sys.argv[0]))
        sys.exit()

    dataset = sys.argv[1]

    RESULT_DIR = CUR_DIR + dataset
    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)

    if not os.path.exists(CUR_DIR):
        os.mkdir(CUR_DIR)

    print("Running Feature Extraction of CIKM'13 baseline ...\n")

    print("\tProcessing question title length ...", end="")
    question_title_len(dataset=dataset)
    print("\tDone!")

    print("\tProcessing user specific features ...", end="")
    user_specific(dataset=dataset)
    print("\tDone!")

    print("\tProcessing questions & answers content ...", end="")
    question_user_content(dataset=dataset)
    print("\tDone!")

    for mode in ["lm.ans", "lm.ask_ans", "lda"]:
        print("\tProcessing features in {}".format(mode), end="")
        question_user_features(dataset=dataset, mode=mode)
        print("\tDone!")

    print("! - All set - !")

