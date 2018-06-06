import sys, os
import numpy as np

try:
    import ujson as json
except:
    import json

data_dir = os.getcwd() + "/data/"
POST_Q = "Posts_Q.json"


def count_no_accepted(dataset):
    """
    Count questions that don't have accepted answers.
    Args:
        dataset -
    """
    infile = data_dir + dataset + "/" + POST_Q
    total = 0
    nacc = []
    with open(infile, "r") as fin:
        for line in fin:
            data = json.loads(line)
            qid = data.get("Id", None)
            acc_id = data.get("AcceptedAnswerId", None)
            total += 1
            if not acc_id:
                nacc.append(qid)
    print("{} questions in total,"
          "{} questions do not have accepted answer"
          .format(total, len(nacc)))

