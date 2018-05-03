"""
    statistics for the experiments
"""

import torch
from torch.autograd import Variable

import sys, os
from data_loader import DataLoader
from model import NeRank

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
          .format(len(nq), len(nu)))

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

def main():
    # TODO: to be finished here
    """
    dataset - the dataset to test
    k - # of records to mark an key-user
    _id - the ID number of the model
    option - the option to do
                [1] Draw tSNE
                [2] Test non-question
    """
    if len(sys.argv) < ? + 1:
        print("Lacking parameters!\n"
              "\tUsage python {} [dataset] [Id] [k] [option]")

    dataset = sys.argv[1]
    _id = int(sys.argv[2])
    k = int(sys.argv[3])
    option = int(sys.argv[4])


    get_rqa_count(dataset)
    key_a_ind = get_frequent_answerers(dataset, k)

    # TODO: load model
    dl = DataLoader(dataset=dataset, id=_id, include_content=False,
                    mp_length=32, mp_coverage=10)
    valid_model = NeRank(embedding_dim=128, vocab_size=dl.user_count,
                         lstm_layers=1, cnn_channel=32, lambda_=1.5)
    model_name = "" # TODO: fill in model names
    valid_model.load_state_dict(torch.load(model_name))

    if option == 1:
        embed_mat = valid_model.au_embeddings
        plot_key_a(key_a_ind, embed_mat, _id)
    elif option == 2:
        pass



