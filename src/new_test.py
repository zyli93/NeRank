import sys, os
import numpy as np





if len(sys.argv) < 3 + 1:
    print("Not enough params",
            "[dataset] [k] [opt]")
    sys.exit()
dataset = sys.argv[1]
k = int(sys.argv[2])
opt = int(sys.argv[3])
DATADIR = os.getcwd() + "/data/parsed/" + dataset + "/"
old_test = DATADIR + "test.txt"
alist_test = DATADIR + "test_q_as.txt"
qacc_file = DATADIR + "Q_ACC_A.txt"

count = {}
with open(qacc_file, "r") as fin:
    lines = fin.readlines()
    for line in lines:
        qid, accaid = line.strip().split()
        count[accaid] = count.get(accaid, 0) + 1

count_list = list(count.items())
count_list.sort(key=lambda x:x[1], reverse=True)
top_a = [x[0] for x in count_list[:101]]

qalist = {}
with open(alist_test, "r") as fin:
    lines = fin.readlines()
    for line in lines:
        l_split = line.strip().split()
        qid = l_split[0]
        alist = l_split[1:]
        qalist[qid] = alist

if opt == 1:
    new_test = DATADIR + "test_" + dataset + "_1.txt"
    with open(old_test, "r") as fin, \
            open(new_test, "w") as fout:
        lines = fin.readlines()
        for line in lines:
            l_split = line.strip().split()
            rid, qid, accaid = l_split[0:3]
            anss = qalist[qid]
            neg_ans = np.random.choice(top_a, size=k-len(anss), replace=False)
            anss += list(neg_ans)
            anss_str = " ".join(anss)
            print("{} {} {} {}".format(rid, qid, accaid, anss_str), file=fout)
elif opt == 2:
    new_test = DATADIR + "test_" + dataset + "_2.txt"
elif opt == 3:
    new_test = DATADIR + "test_" + dataset + "_3.txt"
else:
    print("Unrecognized opt:{}".format(opt), file=sys.stderr)
    sys.exit()


