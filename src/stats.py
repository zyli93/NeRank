"""
    statistics for the experiments
"""

import torch
from torch.autograd import Variable

import sys, os
from data_loader import DataLoader
from model2 import NeRank
from pder import PDER

try:
    import ujson as json
except:
    import json

uset, qset = set(), set()

a_count = {}
key_a = []

PARSED = os.getcwd() + "/data/parsed/"
DATA = os.getcwd() + "/data/"
PLOTDIR = os.getcwd() + "/image/"

def get_rqa_count(dl):
    nq = len(dl.qid2len.keys())
    nu = dl.user_count
    print("Dataset has Question:{} and User:{}"
          .format(nq, nu))

def get_frequent_answerers(dataset, k, dl):
    qaufile = DATA + dataset + "QAU_Map.json"
    with open(qaufile, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            data = json.loads(line)
            alist = data['AnswerOwnerList']
            for _, aid in alist:
                aid = int(aid)
                a_count[aid] = a_count.get(aid, 0) + 1

    for aid, count in a_count.items():
        if count > k:
            key_a.append(aid)

    key_a_ind = [dl.uid2ind[aid] for aid in key_a]
    return key_a_ind

def plot_key_a(key_a_ind, embed_mat, _id):
    key_a_ind = Variable(torch.LongTensor(key_a_ind))
    key_a_emb = embed_mat(key_a_ind).numpy()

    if not os.path.exists(PLOTDIR):
        os.mkdir(PLOTDIR)
    plot_name = PLOTDIR + str(_id) + ".png"
    # Plot tSNE here

def validate(dl, model, test_prop=None):
    model.eval()
    tbatch = dl.build_test_batch(test_prop=test_prop)
    MRR, hit_K, prec_1 = 0, 0, 0
    tbatch_len = len(tbatch)

    # The format of tbatch is:  [aids], rid, qid, accid
    for rid, qid, accid, aid_list in tbatch:
        rank_a = Variable(torch.LongTensor(dl.uid2index(aid_list)))
        rep_rid = [rid] * len(aid_list)
        rank_r = Variable(torch.LongTensor(dl.uid2index(rep_rid)))
        rank_q_len= dl.q2len(qid)
        rank_q = Variable(torch.FloatTensor(dl.q2emb(qid)))

        if torch.cuda.is_available():
            rank_a = rank_a.cuda()
            rank_r = rank_r.cuda()
            rank_q = rank_q.cuda()
        score = model(rpos=None, apos=None, qinfo=None,
                      rank=None, nsample=None, dl=dl,
                      test_data=[rank_a, rank_r, rank_q, rank_q_len], train=False)
        RR, hit, prec = dl.perform_metric(aid_list, score, accid, self.prec_k)
        MRR += RR
        hit_K += hit
        prec_1 += prec

    MRR, hit_K, prec_1 = MRR/tbatch_len, hit_K/tbatch_len, prec_1/tbatch_len
    return MRR, hit_K, prec_1

def main():
    # TODO: to be finished here
    """
    dataset - the dataset to test
    _id - the ID number of the model
    """
    if len(sys.argv) < 3 + 1:
        print("Lacking parameters!\n"
              "\tUsage python {} [dataset] [Id] [model_name]".format(sys.argv[0]))

    dataset = sys.argv[1]
    _id = int(sys.argv[2])
    model_name = sys.argv[3]


    get_rqa_count(dataset)

    dl = DataLoader(dataset=dataset, id=_id, include_content=False,
                    mp_length=32, mp_coverage=10)
    valid_model = NeRank(embedding_dim=128, vocab_size=dl.user_count,
                         lstm_layers=1, cnn_channel=32, lambda_=1.5)
    valid_model.load_state_dict(torch.load(model_name))
    print(validate(dl, valid_model, 1))






