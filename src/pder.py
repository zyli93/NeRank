"""

    Personalized Domain Expert Recommendation

    author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>

"""

import os
import datetime

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn

from model import NeRank
from data_loader import DataLoader


class PDER:
    def __init__(self, dataset, embedding_dim, epoch_num,
                 batch_size, window_size, neg_sample_ratio,
                 lstm_layers, include_content, lr, cnn_channel,
                 test_ratio, lambda_, prec_k,
                 mp_length, mp_coverage, id):

        self.dataset = dataset
        self.dl = DataLoader(dataset=dataset, id=id,
                             include_content=include_content,
                             mp_coverage=mp_coverage, mp_length=mp_length)
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.window_size = window_size
        self.epoch_num = epoch_num
        self.neg_sample_ratio = neg_sample_ratio
        self.lstm_layers = lstm_layers
        self.learning_rate = lr

        self.test_prop = test_ratio
        self.prec_k = prec_k
        self.id = id

        self.model_folder = os.getcwd() + "/model/"

        self.model = NeRank(embedding_dim=self.embedding_dim,
                            vocab_size=self.dl.user_count,
                            lstm_layers=self.lstm_layers,
                            cnn_channel=cnn_channel,
                            lambda_=lambda_)

    def train(self):
        dl = self.dl
        model = self.model
        if torch.cuda.device_count() > 1:
            print("Using {} GPUs".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        if torch.cuda.is_available():  # Check availability of cuda
            print("Using device {}".format(torch.cuda.current_device()))
            model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        batch_count = 0
        best_MRR, best_hit_K, best_pa1 = 0, 0, 0
        for epoch in range(self.epoch_num):
            epoch_total_loss = 0
            dl.process = True
            iter = 0

            while dl.process:
                upos, vpos, npos, nsample, aqr, accqr \
                    = dl.generate_batch(
                        window_size=self.window_size,
                        batch_size=self.batch_size,
                        neg_ratio=self.neg_sample_ratio)

                """
                In order to totally vectorize the computation, 
                we use following method.
                qupos, qvpos, qnpos are the positions and ids of the text.
                qudiag, qvdiag, qndiag have 1 in cells 
                where the corresponding cell in qupos
                    (qvpos, qnpos) is not zero.
                We use dot product to zero out those undesired columns.

                Create the Variables, only Variables with LongTensor can be
                  sent to nn.Embedding
                """

                # R-u, R-v, and R-n
                rupos = Variable(torch.LongTensor(dl.uid2index(upos[0])))
                rvpos = Variable(torch.LongTensor(dl.uid2index(vpos[0])))
                rnpos = Variable(torch.LongTensor(dl.uid2index(npos[0])))
                rpos = [rupos, rvpos, rnpos]

                # A-u, A-v, and A-n
                aupos = Variable(torch.LongTensor(dl.uid2index(upos[1])))
                avpos = Variable(torch.LongTensor(dl.uid2index(vpos[1])))
                anpos = Variable(torch.LongTensor(dl.uid2index(npos[1])))
                apos = [aupos, avpos, anpos]

                qu_wc = dl.qid2vec_padded(upos[2])
                qv_wc = dl.qid2vec_padded(vpos[2])
                qn_wc = dl.qid2vec_padded(npos[2])

                qulen = dl.qid2vec_len(upos[2])
                qvlen = dl.qid2vec_len(vpos[2])
                qnlen = dl.qid2vec_len(npos[2])

                qu_wc = Variable(torch.FloatTensor(qu_wc).view(-1, dl.PAD_LEN, 300))
                qv_wc = Variable(torch.FloatTensor(qv_wc).view(-1, dl.PAD_LEN, 300))
                qn_wc = Variable(torch.FloatTensor(qn_wc).view(-1, dl.PAD_LEN, 300))
                qulen = Variable(torch.LongTensor(qulen))
                qvlen = Variable(torch.LongTensor(qvlen))
                qnlen = Variable(torch.LongTensor(qnlen))

                qinfo = [qu_wc, qv_wc, qn_wc, qulen, qvlen, qnlen]

                # aqr: R, A, Q

                rank_r = Variable(torch.LongTensor(dl.uid2index(aqr[:, 0])))
                rank_a = Variable(torch.LongTensor(dl.uid2index(aqr[:, 1])))
                rank_acc = Variable(torch.LongTensor(dl.uid2index(accqr)))
                rank_q_wc= dl.qid2vec_padded(aqr[:, 2])
                rank_q_len = dl.qid2vec_len(aqr[:, 2])
                rank_q = Variable(torch.FloatTensor(rank_q_wc).view(-1, dl.PAD_LEN, 300))
                rank_q_len = Variable(torch.LongTensor(rank_q_len))

                rank = [rank_r, rank_a, rank_acc, rank_q, rank_q_len]

                if torch.cuda.is_available():
                    rpos = [x.cuda() for x in rpos]
                    apos = [x.cuda() for x in apos]
                    qinfo = [x.cuda() for x in qinfo]
                    rank = [x.cuda() for x in rank]

                optimizer.zero_grad()

                # loss and rank_loss, the later for evaluation
                model.train()
                loss = model(rpos=rpos, apos=apos, qinfo=qinfo,
                             rank=rank, nsample=nsample, dl=dl,
                             test_data=None)
                loss.backward()
                optimizer.step()

                epoch_total_loss += loss.data[0]  # type(loss) = Variable
                iter += 1
                batch_count += 1

                """
                Ideas of printing validation, recording performance, 
                    and dumping model.
                    1 - Ea. 5: print epoch, iter, time 
                    2 - Ea. 20: print iter, MRR, haK, pa1, sampled vald set
                    3 - Ea. 100: record iMRR, haK, pa1, sampled vald set
                    4 - Ea. 1000: check if better result, dump model, all valid set
                    5 - ea. epoch: print, record
                """
                if iter % 10 == 0:
                    print("Epoch-{}, Iter-{}".format(epoch, iter), end=" ")
                    tr = datetime.datetime.now().isoformat()[8:24]
                    print("Size-{} ".format(nsample) + tr, end=" ")
                    print("Loss-{:.3f}".format(loss.data[0]))

                # if iter % 20 == 0:
                #     iMRR, ihit_K, ipa1 = self.__validate(test_prop=self.test_prop)
                #     print("\tSampled validate: iter-{}, MRR={:.4f}, hit_K={:.4f}, pa1={:.4f}"
                #             .format(iter, iMRR, ihit_K, ipa1))
                if batch_count % 10 == 0:
                    hMRR, hhit_K, hpa1 = self.__validate()
                    print("\tEntire validate: iter-{},  MRR={:.4f},  hit_K={:.4f},  pa1={:.4f}"
                            .format(iter, hMRR, hhit_K, hpa1))
                    msg = "{:d},{:d},{:.6f},{:.6f},{:.6f}"\
                            .format(epoch, iter, hMRR, hhit_K, hpa1)
                    dl.write_perf_tofile(msg=msg)

                if iter % 1000 == 0:
                    kMRR, khit_K, kpa1 = self.__validate()
                    if sum([kMRR > best_MRR, khit_K > best_hit_K, kpa1 > best_pa1]) > 1:
                        print("--->better pref: MRR-{:.6f}, hitK-{:.6f}, pa1-{:.6f}".
                                format(kMRR, khit_K, kpa1))
                        best_MRR, best_hit_K, best_pa1 = kMRR, khit_K, kpa1
                        if not os.path.exists(self.model_folder):
                            print("Creating Model folder")
                            os.mkdir(self.model_folder)

                        torch.save(model.state_dict(),
                                   self.model_folder + self.dataset + "_" + \
                                           str(self.id) + "_E{}I{}".format(epoch, iter))


            eMRR, ehit_K, epa1 = self.__validate()
            print("Vald@Epoch-{:d}, MRR-{:.6f}, hit_K-{:.6f}, pa1-{:.6f}"
                  .format(epoch, eMRR, ehit_K, epa1))
            msg = "{:d},{:d},{:.6f},{:.6f},{:.6f}"\
                    .format(epoch, iter, eMRR, ehit_K, epa1)
            dl.write_perf_tofile(msg=msg)


        print("Optimization Finished!")

    def __validate(self, test_prop=None):
        dl = self.dl
        model = self.model
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


    def test(self):
        print("Testing under construction.")
        pass
        # TODO: implement here


if __name__ == "__main__":
    pder = PDER() # TODO: implement here
    pder.train()
    pder.test()
