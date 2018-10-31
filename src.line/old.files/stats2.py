try:
    import ujson as json
except:
    import json

import os, sys

DATA_DIR = os.getcwd() + "/data/{}/".format(sys.argv[1])
setR, setA = set(), set()
with open(DATA_DIR + "Record_All.json", "r") as fin:
    line = fin.readline()
    while line:
        data = json.loads(line)
        try:
            rid = data['QuestionOwnerId']
            aid_list = data['AnswererIdList']
        except:
            pass
        
        setR.update([rid])
        setA.update(aid_list)
        line = fin.readline()

print(len(setR))
print(len(setA))

