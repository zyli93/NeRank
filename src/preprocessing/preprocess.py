#!/usr/bin/env python3

import sys
import os
import re
from lxml import etree
#from xml.etree import cElementTree
from html.parser import HTMLParser
from bs4 import BeautifulSoup

# lxml.html.document_fromstring(html_string)

try:
    import ujson as json
except:
    import json


def clean_html(x):
    return BeautifulSoup(x, 'lxml').get_text()


def preprocess_general(PATH_RAW, PATH_DATA, raw_file, output_file):
    html_parser = HTMLParser()
    parser = etree.iterparse(PATH_RAW + raw_file, events=('end',), tag='row')
    with open(PATH_DATA + output_file, 'w') as fout:
        for event, elem in parser:
            attr = dict(elem.attrib)
            print(json.dumps(attr), file=fout)
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]


def preprocess_posts(PATH_RAW, PATH_DATA):
    html_parser = HTMLParser()
    parser = etree.iterparse(PATH_RAW + 'Posts.xml',
                             events=('end',), tag='row')
    with open(PATH_DATA + 'posts_question.json', 'w') as fout_q, \
            open(PATH_DATA + 'posts_answer.json', 'w') as fout_a:
        for event, elem in parser:
            attr = dict(elem.attrib)
            attr['Body'] = clean_html(attr['Body'])
            if attr['PostTypeId'] == '1':
                print(json.dumps(attr), file=fout_q)
            elif attr['PostTypeId'] == '2':
                print(json.dumps(attr), file=fout_a)

            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]


def preprocess_comments(PATH_RAW, PATH_DATA):
    html_parser = HTMLParser()
    parser = etree.iterparse(PATH_RAW + 'Comments.xml',
                             events=('end',), tag='row')
    with open(PATH_DATA + 'comments.json', 'w') as fout:
        for event, elem in parser:
            attr = dict(elem.attrib)
            attr['Text'] = html_parser.unescape(attr['Text'])
            print(json.dumps(attr), file=fout)
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]


def parse_raw_data(PATH_RAW, PATH_DATA):
    # check the existence of the data raw
    assert(os.path.isdir(PATH_RAW))

    # create the corresponding directory for preprocessed raw
    if not os.path.isdir(PATH_DATA):
        os.makedirs(PATH_DATA)

    # start preprocessing
    print('preprocess posts')
    preprocess_posts(PATH_RAW, PATH_DATA)
    print('preprocess comments')
    preprocess_comments(PATH_RAW, PATH_DATA)
    print('preprocess badges')
    preprocess_general(PATH_RAW, PATH_DATA, 'Badges.xml', 'badges.json')
    print('preprocess postlinks')
    preprocess_general(PATH_RAW, PATH_DATA, 'PostLinks.xml', 'postlinks.json')
    print('preprocess tags')
    preprocess_general(PATH_RAW, PATH_DATA, 'Tags.xml', 'tags.json')
    print('preprocess users')
    preprocess_general(PATH_RAW, PATH_DATA, 'Users.xml', 'users.json')
    print('preprocess votes')
    preprocess_general(PATH_RAW, PATH_DATA, 'Votes.xml', 'votes.json')
    print('preprocess posthistory')
    preprocess_general(PATH_RAW, PATH_DATA,
                       'PostHistory.xml', 'posthistory.json')


def link_questions_answers(PATH_DATA):
    # check the existence of the preprocessed question and answer posts
    assert(os.path.exists(PATH_DATA + 'posts_question.json'))
    assert(os.path.exists(PATH_DATA + 'posts_answer.json'))

    # load questions
    info = {}
    with open(PATH_DATA + 'posts_question.json', 'r') as fin:
        for line in fin:
            data = json.loads(line)
            qid, dt = data['Id'], data['CreationDate']
            aid = data[
                'AcceptedAnswerId'] if 'AcceptedAnswerId' in data else None
            info[qid] = {
                'QuestionId': qid,
                'CreationDate': dt,
                'AcceptedAnswerId': aid,
                'AnswerList': []
            }
    # load answers
    with open(PATH_DATA + 'posts_answer.json', 'r') as fin:
        for line in fin:
            data = json.loads(line)
            try:
                aid, qid = data['Id'], data['ParentId']
                if qid in info:
                    info[qid]['AnswerList'].append(aid)
                else:
                    print('Answer %s belongs to an unkonwn question %s' %
                          (aid, qid), file=sys.stderr)
            except:
                print(data)
                continue
    # order questions
    qlist = sorted(list(info.keys()), key=lambda x: info[x]['CreationDate'])
    # dump results
    with open(PATH_DATA + 'question_answer_mapping.json', 'w') as fout:
        for q in qlist:
            del info[q]['CreationDate']
            print(json.dumps(info[q]), file=fout)


def split_train_test(PATH_DATA):
    # check the existence of the preprocessed question posts
    assert(os.path.exists(PATH_DATA + 'question_answer_mapping.json'))

    # load question-answer mapping
    instances = []
    with open(PATH_DATA + 'question_answer_mapping.json', 'r') as fin:
        for line in fin:
            data = json.loads(line)
            instances.append(data)

    # 50% former instances are training raw
    num_train = len(instances) // 2

    # create an ID set for training posts
    train_id = set()
    for i in range(num_train):
        train_id.add(instances[i]['QuestionId'])

    # dump train/test mappings
    with open(PATH_DATA + 'train.question_answer_mapping.json', 'w') as fout:
        for i in range(num_train):
            if instances[i]['AcceptedAnswerId'] != None and len(instances[i]['AnswerList']) > 1:
                print(json.dumps(instances[i]), file=fout)
    with open(PATH_DATA + 'test.question_answer_mapping.json', 'w') as fout:
        for i in range(num_train, len(instances)):
            if instances[i]['AcceptedAnswerId'] != None and len(instances[i]['AnswerList']) > 1:
                print(json.dumps(instances[i]), file=fout)

    # dump train/test question and answer posts
    with open(PATH_DATA + 'posts_question.json', 'r') as fin, \
            open(PATH_DATA + 'train.posts_question.json', 'w') as fout_train, \
            open(PATH_DATA + 'test.posts_question.json', 'w') as fout_test:
        for line in fin:
            data = json.loads(line)
            print(line, end='', file=fout_train if data[
                  'Id'] in train_id else fout_test)

    with open(PATH_DATA + 'posts_answer.json', 'r') as fin, \
            open(PATH_DATA + 'train.posts_answer.json', 'w') as fout_train, \
            open(PATH_DATA + 'test.posts_answer.json', 'w') as fout_test:
        for line in fin:
            data = json.loads(line)
            print(line, end='', file=fout_train if data[
                  'ParentId'] in train_id else fout_test)


if __name__ == '__main__':
    if len(sys.argv) < 1 + 1:
        print('--usage %s name_of_the_dataset' % sys.argv[0], file=sys.stderr)
        sys.exit(0)

    # load
    dataset = sys.argv[1]
    PATH_RAW = '../../data/%s/' % dataset
    PATH_DATA = '../../raw/%s/' % dataset

    print('Parsing the data dataset...', file=sys.stderr)
    parse_raw_data(PATH_RAW, PATH_DATA)

    print('Linking questions and corresponding answers...', file=sys.stderr)
    link_questions_answers(PATH_DATA)

    print('Spliting the dataset into training and testing datasets...', file=sys.stderr)
    split_train_test(PATH_DATA)
