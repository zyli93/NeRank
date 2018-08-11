try:
    import ujson as json
except:
    import json

import os

DATASET = "superuser"

DATADIR = os.getcwd() + "/data/{}/".format(DATASET)

POST_Q = "Posts_Q.json"
POST_A = "Posts_A.json"

countQ = {}
countA = {}

with open(DATADIR + "norecord.txt", "w") as fout:
    with open(DATADIR + POST_Q, "r") as fin_q:
        for line in fin_q:
            data = json.loads(line)
            qid, owner_id = data.get("Id", None), data.get("OwnerUserId", None)
            countQ[qid] = countQ.get(qid, 0) + 1
            if not qid or not owner_id:
                print("qid {}, owner_id {}".format(qid, owner_id), file=fout)

    with open(DATADIR + POST_A, "r") as fin_a:
        for line in fin_a:
            data = json.loads(line)
            aid, owner_id = data.get("Id", None), data.get("OwnerUserId", None)
            countA[aid] = countA.get(aid, 0) + 1
            if not aid or not owner_id:
                print("aid {}, owner_id {}".format(aid, owner_id), file=fout)

qlist = sorted(list(countQ.keys()), key=lambda x: countQ[x])
alist = sorted(list(countA.keys()), key=lambda x: countA[x])

with open(DATADIR + "data_stats.txt", "w") as fout:
    for q in qlist:
        print("q: {}, count: {}".format(q, countQ[q]), file=fout)
    for a in alist:
        print("a: {}, count: {}".format(a, countA[a]), file=fout)

