"""
    Random walk generator

    Author:
        Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@ucla.edu>

    Description:
        Generating random walks on our Uq, Ua, and Q network using NetworkX.


"""

import os, sys
import networkx as nx
import random
import numpy as np
import math

from collections import Counter
import itertools



class MetaPathGenerator:
    """MetaPathGenerator

    Args:
        dataset     - the dataset to work on
        length      - the length of random walks to be generated
        num_walks   - the number of random walks start from each node
    """

    def __init__(self, dataset, length=100, coverage=10000):
        self._walk_length = length
        self._coverage = coverage
        self._dataset = dataset
        self.G = nx.Graph()

        self.walks = []
        self.pairs = []

        self.initialize()

    def initialize(self):
        """ Initialize Graph

        Initialize graph with Uq-Q pairs and Q-Ua pairs.
        We use following Uppercase letter

        Args:
            QR_file - Input file containing Q-R pairs
            QA_file - Input file containing Q-A pairs

        """

        DATA_DIR = os.getcwd() + "/data/parsed/" + self._dataset + "/"
        QR_file = DATA_DIR + "Q_R.txt"
        QA_file = DATA_DIR + "Q_A.txt"
        G = self.G
        # Read in Uq-Q pairs
        with open(QR_file, "r") as fin:
            lines = fin.readlines()
            RQ_edge_list = []
            for line in lines:
                unit = line.strip().split()
                RQ_edge_list.append(["Q_" + unit[0],
                                     "R_" + unit[1]])
            G.add_edges_from(RQ_edge_list)
        with open(QA_file, "r") as fin:
            lines = fin.readlines()
            QA_edge_list = []
            for line in lines:
                unit = line.strip().split()
                QA_edge_list.append(["Q_" + unit[0],
                                     "A_" + unit[1]])
            G.add_edges_from(QA_edge_list)

    def get_nodelist(self, type=None):
        """ Get specific type or all nodes of nodelist in the graph

        Args:
            type - The entity type of the entity.
                   If set as `None`, then all types of nodes would be returned.

        Return:
            nodelist - the list of node with `type`
        """
        G = self.G

        if not G.number_of_edges() or not G.number_of_nodes():
            sys.exit("Graph should be initialized before get_nodelist()!")

        if not type:
            return list(G.nodes)
        return [node for node in list(G.nodes)
                if node[0] == type]

    def generate_metapaths(self, patterns, alpha):
        """ Generate Random Walk

        Generating random walk from the Tripartite graph
        A candidate pattern pool is:
            "A-Q-R-Q-A": specifies 2 A's answered a question proposed by a same R
            "A-Q-A": speficies 2 A' answered a same question

        Args:
            meta_pattern - the pattern that guides the walk generation
            alpha - probability of restart

        Return:
            walks - a set of generated random walks
        """
        G = self.G
        num_walks, walk_len = self._coverage, self._walk_length
        rand = random.Random(0)

        print("Generating Meta-paths ...")

        if not G.number_of_edges() or not G.number_of_nodes():
            sys.exit("Graph should be initialized before generate_walks()!")

        walks = []

        for meta_pattern in patterns:  # Generate by patterns
            print("\tNow generating meta-paths from pattern: \"{}\" ..."
                  .format(meta_pattern))
            start_entity_type = meta_pattern[0]
            start_node_list = self.get_nodelist(start_entity_type)
            for cnt in range(num_walks):  # Iterate the node set for cnt times
                print("Count={}".format(cnt))
                rand.shuffle(start_node_list)
                total = len(start_node_list)                
                for ind, start_node in enumerate(start_node_list):
                    if ind % 3000 == 0:
                        print("Finished {:.2f}".format(ind/total))

                    walks.append(
                        self.__meta_path_walk(
                            start=start_node,
                            alpha=alpha,
                            pattern=meta_pattern))

        print("Done!")
        self.walks = walks
        return

    def generate_metapaths_2(self):
        """ Generate Random Walk

        Generating random walk from the Tripartite graph
        Args:
            meta_pattern - the pattern that guides the walk generation
            alpha - probability of restart

        Return:
            walks - a set of generated random walks
        """
        G = self.G
        num_walks, walk_len = self._coverage, self._walk_length
        rand = random.Random(0)

        print("Generating Meta-paths ...")

        if not G.number_of_edges() or not G.number_of_nodes():
            sys.exit("Graph should be initialized before generate_walks()!")

        walks = []

        print("\tNow generating meta-paths from deepwalk ...")
        start_node_list = self.get_nodelist()
        for cnt in range(num_walks):  # Iterate the node set for cnt times
            print("Count={}".format(cnt))
            rand.shuffle(start_node_list)
            total = len(start_node_list)
            for ind, start_node in enumerate(start_node_list):
                if ind % 3000 == 0:
                    print("Finished {:.2f}".format(ind/total))
                walks.append(
                    self.__random_walk(start=start_node))

        print("Done!")
        self.walks = walks
        return

    def __random_walk(self, start=None):
        """Single Random Walk Generator

        Args:
            rand - an random object to generate random numbers
            start - starting node

        Return:
            walk - the single walk generated
        """
        G = self.G
        rand = random.Random()
        walk = [start]
        cur_node = start
        while len(walk) <= self._walk_length:
            possible_next_nodes = [neighbor
                                   for neighbor in G.neighbors(cur_node)]
            next_node = rand.choice(possible_next_nodes)
            walk.append(next_node)
            cur_node = next_node

        return " ".join(walk)

    def __meta_path_walk(self, start=None, alpha=0.0, pattern=None):
        """Single Walk Generator

        Generating a single random walk that follows a meta path of `pattern`

        Args:
            rand - an random object to generate random numbers
            start - starting node
            alpha - probability of restarts
            pattern - (string) the pattern according to which to generate walks
            walk_len - (int) the length of the generated walk

        Return:
            walk - the single walk generated

        """
        def type_of(node_id):
            return node_id[0]

        rand = random.Random()
        # Checking pattern is correctly initialized
        if not pattern:
            sys.exit("Pattern is not specified when generating meta-path walk")

        G = self.G
        n, pat_ind = 1, 1

        walk = [start]

        cur_node = start

        # Generating meta-paths
        while len(walk) <= self._walk_length or pat_ind != len(pattern):

            # Updating the pattern index
            pat_ind = pat_ind if pat_ind != len(pattern) else 1

            # Decide whether to restart
            if rand.random() >= alpha:
                # Find all possible next neighbors
                possible_next_node = [neighbor
                                      for neighbor in G.neighbors(cur_node)
                                      if type_of(neighbor) == pattern[pat_ind]]
                # Random choose next node
                next_node = rand.choice(possible_next_node)
            else:
                next_node = walk[0]

            walk.append(next_node)
            cur_node = next_node
            pat_ind += 1

        return " ".join(walk)

    def write_metapaths(self):
        """Write Metapaths to files

        Args:
            walks - The walks generated by `generate_walks`
        """

        print("Writing Generated Meta-paths to files ...", end=" ")

        DATA_DIR = os.getcwd() + "/metapath/"
        OUTPUT = DATA_DIR + self._dataset + "_" \
                 + str(self._coverage) + "_" + str(self._walk_length) + ".txt"
        if not os.path.exists(DATA_DIR):
            os.mkdir(DATA_DIR)
        with open(OUTPUT, "w") as fout:
            for walk in self.walks:
                print("{}".format(walk), file=fout)

        print("Done!")

    def path_to_pairs(self, window_size):
        """Convert all metapaths to pairs of nodes

        Args:
            walks - all the walks to be translated
            window_size - the sliding window size
        Return:
            pairs - the *shuffled* pair corpus of the dataset
        """
        pairs = []
        if not self.walks:
            sys.exit("Walks haven't been created.")
        for walk in self.walks:
            walk = walk.strip().split(' ')
            for pos, token in enumerate(walk):
                lcontext, rcontext = [], []
                lcontext = walk[pos - window_size: pos] \
                    if pos - window_size >= 0 \
                    else walk[:pos]

                if pos + 1 < len(walk):
                    rcontext = walk[pos + 1: pos + window_size] \
                        if pos + window_size < len(walk) \
                        else walk[pos + 1:]

                context_pairs = [[token, context]
                                 for context in lcontext + rcontext]
                pairs += context_pairs
        np.random.shuffle(pairs)
        self.pairs = pairs
        return

    def write_pairs(self):
        """Write all pairs to files
        Args:
            pairs - the corpus
        Return:
        """
        print("Writing Generated Pairs to files ...")
        DATA_DIR = os.getcwd() + "/corpus_line/"
        OUTPUT = DATA_DIR + self._dataset + "_" + \
                 str(self._coverage) + "_" + str(self._walk_length) + ".txt"
        if not os.path.exists(DATA_DIR):
            os.mkdir(DATA_DIR)
        with open(OUTPUT, "w") as fout:
            for pair in self.pairs:
                print("{} {}".format(pair[0], pair[1]), file=fout)
        return

    def down_sample(self):
        """Down sampling the training sets
        
        1. Remove all the duplicate tuples such as "A_11 A_11"
        2. Take log of all tuples as a down sampling
        """

        pairs = self.pairs
        pairs = [(pair[0], pair[1])
                 for pair in pairs
                 if pair[0] != pair[1]]
        cnt = Counter(pairs)
        down_cnt = [[pair] * math.ceil(math.log(count))
                    for pair, count in cnt.items()]
        self.pairs = list(itertools.chain(*down_cnt))
        np.random.shuffle(self.pairs)

    def sample_LINE(self):
        pairs = self.pairs
        for u, v in self.G.edges:
            pairs.append((u, v))
            pairs.append((v, u))


if __name__ == "__main__":
    if len(sys.argv) < 4 + 1:
        print("\t Usage:{} "
              "[name of dataset] [length] [num_walk] [window_size]"
              .format(sys.argv[0], file=sys.stderr))
        sys.exit(1)
    dataset = sys.argv[1]
    length = int(sys.argv[2])
    num_walk = int(sys.argv[3])
    window_size = int(sys.argv[4])
    
    gw = MetaPathGenerator(length=length, coverage=num_walk, dataset=dataset)

    # Uncomment the first line for metapath-based

    # gw.generate_metapaths(patterns=["AQRQA"], alpha=0)
    # gw.generate_metapaths_2()

    # gw.path_to_pairs(window_size=window_size)
    # gw.down_sample()

    # gw.write_metapaths()

    gw.sample_LINE()

    gw.write_pairs()




