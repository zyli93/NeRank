#!/home/zeyu/anaconda3/bin/python3.6

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


def split_post(raw_dir, data_dir):
    """
    Split post to question and answer,
    keep all information, output to file

    :param raw_dir: raw data directory
    :param data_dir: parsed data directory
    :return: nothing
    """
    print("Splitting Questions and Answers ...")
    with open(data_dir + "Posts_Q.json", "w") as fout_q, \
        open(data_dir + "Posts_A.json", "w") as fout_a:
        parser = etree.iterparse(raw_dir + 'Posts.xml',
                                 events=('end',), tag='row')
        for event, elem in parser:
            attr = dict(elem.attrib)
            attr['Body'] = clean_html(attr['Body'])

            # Output to separate files
            if attr['PostTypeId'] == '1':
                fout_q.write(json.dumps(attr) + "\n")
            elif attr['PostTypeId'] == '2':
                fout_a.write(json.dumps(attr) + "\n")
    print("Done! Generated {} and {}."
          .format(data_dir + "Posts_Q.json", data_dir + "Posts_A.json"))
    return


def process_QA(data_dir):
    """
    Process QA, extract useful information.
    Get rid of the text information,
    only record the question-user - answer-user relation

    :param data_dir: the dir where primitive data is stored
    :return:
    """
    print("Processing QA relations ...")

    POST_Q = "Posts_Q.json"
    POST_A = "Posts_A.json"
    OUTPUT = "QAU_Map.json"

    if not os.path.exists(data_dir + POST_Q):
        raise IOError("file {} does NOT exist".format(data_dir + "Posts_Q.json"))

    if not os.path.exists(data_dir + POST_A):
        raise IOError("file {} does NOT exist".format(data_dir + "Posts_A.json"))

    qa_map = {}

    # Process question information
    with open(data_dir + POST_Q, 'r') as fin_q:
        for line in fin_q:
            data = json.loads(line)
            try:
                qid, owner_id = data['Id'], data['OwnerUserId']

                acc_id = data.get('AcceptedAnswerId', None)

                qa_map[qid] = {
                    'QuestionId': qid,
                    'QuestionOwnerId': owner_id,
                    'AcceptedAnswerId': acc_id,
                    'AnswerOwnerList': []
                }
            except:
                # TODO: handle exceptional data
                print(data)
                continue

    # Process answer information
    with open(data_dir + POST_A, 'r') as fin_a:
        for line in fin_a:
            data = json.loads(line)
            try:
                aid, owner_id = data['Id'], data['OwnerUserId']

                par_id = data['ParentId']

                entry = qa_map.get(par_id, None)
                if entry:
                    entry['AnswerOwnerList'].append((aid, owner_id))

                else:
                    print("Answer {} belongs to unknown Question {}"
                          .format(aid, par_id), file=sys.stderr)
            except:
                # TODO: handle exceptional data
                print(data)
                continue

    # Sort qid list, write to file by order of qid
    qid_list = sorted(list(qa_map.keys()),
                      key= lambda x: qa_map[x]['QuestionId'])

    # Write QA pair to file
    with open(data_dir + OUTPUT, 'w') as fout:
        for q in qid_list:
            fout.write(json.dumps(qa_map[q]) + "\n")
    print("Done! Question-Answer-User relationships are stored in"
          "{}".format(data_dir + OUTPUT))


def extract_question_user(data_dir, parsed_dir):
    """
    Extract Question User pairs and output to file.
    Format: <qid> <uid>
    E.g.
    101 40
    145 351

    :param data_dir: data directory
    :param parsed_dir: parsed file directory
    :return:
    """
    print("Extracting Question User pairs ...")
    INPUT = "QAU_Map.json"
    OUTPUT = "q_u.txt"

    if not os.path.exists(data_dir + INPUT):
        IOError("Can NOT find {}".format(data_dir + INPUT))

    with open(data_dir + INPUT, "r") as fin:
        with open(parsed_dir + OUTPUT, "w") as fout:
            for line in fin:
                data = json.loads(line)
                qid = data['QuestionId']
                owner_id = data['QuesionOwnerId']
                print("{} {}".format(str(qid), str(owner_id)),
                      file=fout)
    print("Done! Question-User pairs are in {}}"
          .format(parsed_dir + OUTPUT))


def extract_question_answer(data_dir, parsed_dir):
    """
    Extract Question Answer, Answer User pairs and output to file.
    The list of AnswerOwnerList contains <aid>-<owner_id> pairs

    QA pair file Format: <qid> <aid>
    AU pair file Format: <aid> <owner_id>

    E.g.
    101 40
    145 351

    :param data_dir: data directory
    :param parsed_dir: parsed file directory
    :return:
    """
    INPUT = "QAU_Map.json"
    OUTPUT_QA = "q_a.txt"
    OUTPUT_AU = "a_u.txt"

    if not os.path.exists(data_dir + INPUT):
        IOError("Can NOT find {}".format(data_dir + INPUT))

    with open(data_dir + INPUT, "r") as fin:
        with open(parsed_dir + OUTPUT_QA, "w") as fout_qa:
            with open(parsed_dir + OUTPUT_AU, "w") as fout_au:
                for line in fin:
                    data = json.loads(line)
                    qid = data['QuestionId']
                    au_list = data['AnswerOwnerList']
                    for aid, owner_id in au_list:
                        print("{} {}".format(str(qid), str(aid)),
                              file=fout_qa)
                        print("{} {}".format(str(aid), str(owner_id)),
                              file=fout_au)

def extract_question_content(data_dir, parsed_dir):
    # TODO: start from here


if __name__ == "__main__":
    if len(sys.argv) > 1 + 1:
        print("\t Usage: {} [name of dataset]"
            .format(sys.argv[0]), file=sys.stderr)
        sys.exit(0)

    DATASET = sys.argv[1]
    RAW_DIR = "./raw/{}/".format(DATASET)
    DATA_DIR= "./data/{}/".format(DATASET)
    PARSED_DIR = "./data/parsed/{}/".format(DATASET)
    README_FILE = "./data/README.txt"


    if not os.path.exists(RAW_DIR):
        print("{} dir or path doesn't exist.\n"
              "Please download the raw data set into the /raw."
              .format(RAW_DIR), file=sys.stderr)

    if not os.path.exists(DATA_DIR):
        print("{} data dir not found.\n"
              " Creating a folder for that."
              .format(DATA_DIR))
        os.makedirs(DATA_DIR)

    if not os.path.exists(PARSED_DIR):
        print("{} dir or path NOT found.\n"
              "Creating a foler for that."
              .format(PARSED_DIR))
        os.makedirs(PARSED_DIR)


    print("******************************************")
    split_post(raw_dir=RAW_DIR, data_dir=DATA_DIR)

    # extract question-user, answer-user, and question-answer information
    process_QA(data_dir=DATA_DIR)

    print("Extracting Uq - Q pairs ...")
    extract_question_user(data_dir=DATA_DIR, pased_dir=PARSED_DIR)

    # print("Extracting Q - Ua pairs ...")

    # print("Extracting Q - Qcontent pairs ...")

    # print("Creating Uq Q Ua* Ua tuples...")

    print("Done!")


