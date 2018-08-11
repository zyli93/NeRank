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

from embed import Embed
from skipgram import SkipGram
from recsys import RecSys
from data_loader import DataLoader
from utils import Utils


class PDER:
    def __init__(self, dataset, embedding_dim, epoch_num,
                 batch_size, neg_sample_ratio,
                 lstm_layers, include_content, lr, cnn_channel,
                 test_ratio, lambda_, prec_k,
                 mp_length, mp_coverage, id, answer_sample_ratio):

        self.dataset = dataset
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.neg_sample_ratio = neg_sample_ratio
        self.lstm_layers = lstm_layers
        self.learning_rate = lr
        self.test_prop = test_ratio
        self.prec_k = prec_k
        self.id = id

        self.dl = DataLoader(dataset=dataset
                             , ID=id
                             , include_content=include_content
                             , coverage=mp_coverage
                             , length=mp_length
                             , answer_sample_ratio=answer_sample_ratio
                             )

        self.utils = Utils(dataset=dataset
                           , ID=id
                           , mp_coverage=mp_coverage
                           , mp_length=mp_length
                           )

        self.model_folder = os.getcwd() + "/model/"

        print(self.dl.user_count)
        self.embedding_manager = Embed(vocab_size=self.dl.user_count + 1
                                       , embedding_dim=embedding_dim
                                       , lstm_layers=lstm_layers
                                       )

        self.skipgram = SkipGram(embedding_dim=self.embedding_dim
                                 , emb_man=self.embedding_manager
                                 )

        self.recsys = RecSys(embedding_dim=embedding_dim
                             , cnn_channel=cnn_channel
                             , embeddings=self.embedding_manager
                             )

    def run(self):
        dl, utils = self.dl, self.utils
        recsys, skipgram = self.recsys, self.skipgram

        if torch.cuda.device_count() > 1:
            print("Using {} GPUs".format(torch.cuda.device_count()))
            skipgram = nn.DataParallel(skipgram)
            recsys = nn.DataParallel(recsys)

        if torch.cuda.is_available():  # Check availability of cuda
            print("Using device {}".format(torch.cuda.current_device()))
            skipgram.cuda()
            recsys.cuda()

        skipgram_optimizer = optim.Adam(skipgram.parameters()
                                        , lr=self.learning_rate)
        recsys_optimizer = optim.Adam(recsys.parameters()
                                      , lr=0.07 * self.learning_rate)

        batch_count = 0
        best_MRR, best_hit_K, best_pa1 = 0, 0, 0

        for epoch in range(self.epoch_num):
            dl.process = True
            iter = 0

            while dl.process:
                upos, vpos, npos, aqr, accqr \
                    = dl.get_train_batch(
                        batch_size=self.batch_size,
                        neg_ratio=self.neg_sample_ratio)

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

                qu_wc = dl.qid2padded_vec(upos[2])
                qv_wc = dl.qid2padded_vec(vpos[2])
                qn_wc = dl.qid2padded_vec(npos[2])

                qulen = dl.qid2vec_length(upos[2])
                qvlen = dl.qid2vec_length(vpos[2])
                qnlen = dl.qid2vec_length(npos[2])

                qu_wc = Variable(torch.FloatTensor(qu_wc).view(-1, dl.PAD_LEN, 300))
                qv_wc = Variable(torch.FloatTensor(qv_wc).view(-1, dl.PAD_LEN, 300))
                qn_wc = Variable(torch.FloatTensor(qn_wc).view(-1, dl.PAD_LEN, 300))
                qulen = Variable(torch.LongTensor(qulen))
                qvlen = Variable(torch.LongTensor(qvlen))
                qnlen = Variable(torch.LongTensor(qnlen))

                qinfo = [qu_wc, qv_wc, qn_wc, qulen, qvlen, qnlen]

                # aqr: R, A, Q
                # print(aqr.shape)
                rank_r = Variable(torch.LongTensor(dl.uid2index(aqr[:, 0])))
                rank_a = Variable(torch.LongTensor(dl.uid2index(aqr[:, 1])))
                rank_acc = Variable(torch.LongTensor(dl.uid2index(accqr)))
                rank_q_wc = dl.qid2padded_vec(aqr[:, 2])
                rank_q_len = dl.qid2vec_length(aqr[:, 2])
                rank_q = Variable(torch.FloatTensor(rank_q_wc).view(-1, dl.PAD_LEN, 300))
                rank_q_len = Variable(torch.LongTensor(rank_q_len))

                rank = [rank_r, rank_a, rank_acc, rank_q, rank_q_len]

                if torch.cuda.is_available():
                    rpos = [x.cuda() for x in rpos]
                    apos = [x.cuda() for x in apos]
                    qinfo = [x.cuda() for x in qinfo]
                    rank = [x.cuda() for x in rank]

                """
                ============== Skip-gram ===============
                """
                skipgram_optimizer.zero_grad()
                skipgram.train()
                skipgram_loss = skipgram(rpos=rpos
                                         , apos=apos
                                         , qinfo=qinfo)
                skipgram_loss.backward()
                skipgram_optimizer.step()

                """
                ============== Rec-Sys ===============
                """
                recsys_optimizer.zero_grad()
                recsys.train()
                recsys_loss = recsys(rank=rank)
                recsys_loss.backward()
                recsys_optimizer.step()

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
                n_sample = upos.shape[1]

                # Print training progress every 10 iterations
                if iter % 10 == 0:
                    tr = datetime.datetime.now().isoformat()[8:24]
                    print("E:{}, I:{}, size:{}, {}, Loss:{:.3f}"
                          .format(epoch, iter, n_sample, tr, skipgram_loss.data[0]))

                # Write to file every 10 iterations
                if batch_count % 10 == 0:
                    # hMRR, hhit_K, hpa1 = 0, 0, 0
                    hMRR, hhit_K, hpa1 = self.test()
                    print("\tEntire Val@ I:{}, MRR={:.4f}, hitK={:.4f}, pa1={:.4f}"
                          .format(iter, hMRR, hhit_K, hpa1))
                    msg = "{:d},{:d},{:.6f},{:.6f},{:.6f}"\
                          .format(epoch, iter, hMRR, hhit_K, hpa1)
                    utils.write_performance(msg=msg)

                # Write to disk every 1000 iterations
                if batch_count % 500 == 0:
                    kMRR, khit_K, kpa1 = self.test()
                    if sum([kMRR > best_MRR, khit_K > best_hit_K, kpa1 > best_pa1]) > 1:
                        print("\t--->Better Pref: MRR={:.6f}, hitK={:.6f}, pa1={:.6f}"
                              .format(kMRR, khit_K, kpa1))
                        best_MRR, best_hit_K, best_pa1 = kMRR, khit_K, kpa1
                        utils.save_model(model=skipgram, epoch=epoch, iter=iter)

            eMRR, ehit_K, epa1 = self.test()
            print("Entire Val@ E:{:d}, MRR-{:.6f}, hit_K-{:.6f}, pa1-{:.6f}"
                  .format(epoch, eMRR, ehit_K, epa1))
            msg = "{:d},{:d},{:.6f},{:.6f},{:.6f}"\
                  .format(epoch, iter, eMRR, ehit_K, epa1)
            utils.write_performance(msg=msg)

        print("Optimization Finished!")

    def test(self, test_prop=None):
        model, dl = self.recsys, self.dl
        model.eval()
        MRR, hit_K, prec_1 = 0, 0, 0

        test_batch = dl.get_test_batch(test_prop=test_prop)
        test_batch_len = len(test_batch)

        # The format of test_batch is:  [aids], rid, qid, accid
        for rid, qid, accaid, aid_list in test_batch:
            rank_a = Variable(torch.LongTensor(dl.uid2index(aid_list)))
            rep_rid = [rid] * len(aid_list)
            rank_r = Variable(torch.LongTensor(dl.uid2index(rep_rid)))
            rank_q_len = dl.q2len(qid)
            rank_q = Variable(torch.FloatTensor(dl.q2emb(qid)))

            if torch.cuda.is_available():
                rank_a = rank_a.cuda()
                rank_r = rank_r.cuda()
                rank_q = rank_q.cuda()

            model.eval()
            score = model.test(test_data=[rank_a, rank_r, rank_q, rank_q_len])
            RR, hit, prec = self.utils.performance_metrics(aid_list
                                                           , score
                                                           , accaid
                                                           , self.prec_k)
            MRR += RR
            hit_K += hit
            prec_1 += prec

        MRR, hit_K, prec_1 = MRR / test_batch_len, hit_K / test_batch_len, prec_1 / test_batch_len
        return MRR, hit_K, prec_1


if __name__ == "__main__":
    pder = PDER()
    pder.run()
