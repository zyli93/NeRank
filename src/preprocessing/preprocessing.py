#!/Users/SuperBlee/anaconda3/bin

import sys
import os
from html.parser import HTMLParser
from lxml import etree
from bs4 import BeautifulSoup

try:
    import ujson as json
except:
    import json

def clean_html(x):
    return BeautifulSoup(x, 'lxml').get_text()


def split_post(RAW_DIR, DATA_DIR):
    # Split post to question and answer,
    # keep all information, output to file
    with open(DATA_DIR + "Posts_Q.json", "w") as fout_q, \
        open(DATA_DIR + "Posts_A.json", "w") as fout_a:
        parser = etree.iterparse(RAW_DIR + 'Posts_copy.xml',
                                 events=('end',), tag='row')
        for event, elem in parser:
            attr = dict(elem.attrib)
            attr['Body'] = clean_html(attr['Body'])
            # Output to separate files
            if attr['PostTypeId'] == '1':
                fout_q.write(json.dumps(attr) + "\n")
            elif attr['PostTypeId'] == '2':
                fout_a.write(json.dumps(attr) + "\n")


def process_QA(DATA_DIR):
    # Process QA, extract useful informations
    assert(os.path.exists(DATA_DIR + "Posts_Q.json"))
    assert(os.path.exists(DATA_DIR + "Posts_A.json"))

    qa_pair = {}

    # Process question information
    with open(DATA_DIR + 'Posts_Q.json', 'r') as fin_q:
        for line in fin_q:
            data = json.loads(line)
            try:
                qid, uq_id = data['Id'], data['OwnerUserId']
                acc_id = data['AcceptedAnswerId'] \
                    if 'AcceptedAnswerId' in data else None
                qa_pair[qid] = {
                    'QuestionId': qid,
                    'QuestionOwnerId': uq_id,
                    'AcceptedAnswerId': acc_id,
                    'UserAnswerTupleList': []
                }
            except:
                print(data)
                continue

    # Process answer information
    with open(DATA_DIR + 'Posts_A.json', 'r') as fin_a:
        for line in fin_a:
            data = json.loads(line)
            try:
                aid, ua_id = data['Id'], data['OwnerUserId']
                pid = data['ParentId']
                if pid in qa_pair:
                    qa_pair[pid]['UserAnswerTupleList'].append((ua_id, aid))
                else:
                    print("Answer {} belongs to unknown Question {}"
                          .format(aid, pid), file=sys.stderr)
            except:
                print(data)
                continue

    q_list = sorted(list(qa_pair.keys()),
                    key= lambda x: qa_pair[x]['QuestionId'])

    # Write question answer to file
    with open(DATA_DIR + 'Parsed_QA.json', 'w') as fout:
        for q in q_list:
            fout.write(json.dumps(qa_pair[q]) + "\n")


def extract_question_user(DATA_DIR):
    assert(os.path.exists(DATA_DIR + "Parsed_QA.json"))
    with open(DATA_DIR + "parsed_QA.json", "r") as fin:
        for line in fin:
            data = json.loads(line)
            qid = data['QuestionId']
            quid = data['QuesionOwnerId']





if __name__ == "__main__":
    if len(sys.argv) > 1 + 1:  # CMD checking
        print("\t Usage: {} [name of dataset]"
            .format(sys.argv[0]), file=sys.stderr)
        sys.exit(0)

    DATASET = sys.argv[1]
    RAW_DIR = "./raw/{}/".format(DATASET)
    DATA_DIR = "./data/{}/".format(DATASET)

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    print("Extracting all Question - Answer pairs ...")

    # split quesitions and answers
    split_post(RAW_DIR, DATA_DIR)

    # extract question-user, answer-user, and question-answer information
    process_QA(DATA_DIR)

    print("Extracting Uq - Q pairs ...")
    extract_question_user(DATA_DIR)

    # print("Extracting Q - Ua pairs ...")

    # print("Extracting Q - Qcontent pairs ...")

    # print("Creating Uq Q Ua* Ua tuples...")

    print("Done!")


