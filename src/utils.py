import os, sys

class Utils:
    def __init__(self, dataset, ID, mp_length, mp_coverage):
        self.dataset = dataset
        self.id = ID

        self.PERF_DIR = os.getcwd() + "/performance/"
        self.performance_file = self.PERF_DIR + \
                                "{}_{}_{}_{}.txt".format(self.dataset, str(self.id),
                                                         str(mp_length), str(mp_coverage))
        pass

    def performance_metrics(self, aid_list, score_list, accid, k):
        """
        Performance metric evaluation

        Args:
            aid_list  -  the list of aid in this batch
            score_list  -  the list of score of ranking
            accid  -  the ground truth
            k  -  precision at K
        """
        if len(aid_list) != len(score_list):
            print("aid_list and score_list not equal length.",
                  file=sys.stderr)
            sys.exit()
        id_score_pair = list(zip(aid_list, score_list))
        id_score_pair.sort(key=lambda x: x[1], reverse=True)
        for ind, (aid, score) in enumerate(id_score_pair):
            if aid == accid:
                if ind == 0:
                    return 1/(ind+1), int(ind < k), 1
                else:
                    return 1/(ind+1), int(ind < k), 0

    def save_model(self):
        pass

    def write_performance(self, msg):
        if not os.path.exists(self.PERF_DIR):
            os.mkdir(self.PERF_DIR)
            with open(self.performance_file,"w") as fout:
                print("Epoch,Iter,MRR,hit_K,pa1", file=fout)
        with open(self.performance_file, "a") as fout:
            print(msg, file=fout)