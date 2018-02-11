"""Random walk generator

Generating random walks on our Uq, Ua, and Q network using NetworkX.

Author: Zeyu Li zyli@cs.ucla.edu

"""

import os, sys, argparse
import networkx as nx
import random


class RandomWalkGenerator:
    """RandomWalkGenerator

    Args:
        length      - the length of random walks to be generated
        num_walks   - the number of random walks start from each node


    """

    def __init__(self, length=100, num_walks=10000):
        self._walk_length = length
        self._num_walks = num_walks
        self.G = nx.Graph()


    def initialize(self, uqq_file, qua_file):
        """ Initialize Graph

        Initialize graph with Uq-Q pairs and Q-Ua pairs.
        We use following Uppercase letter

        Args:
            uqq_file - Input file containing Uq-Q pairs
            qua_file - Input file containing Q-Uq pairs

        """
        G = self.G

        # Read in Uq-Q pairs
        with open(uqq_file, "r") as fin:
            lines = fin.readlines()
            RQ_edge_list = []
            for line in lines:
                unit = line.strip().split()
                RQ_edge_list.append(["R_" + unit[0],
                                     "Q_" + unit[1]])
            G.add_edges_from(RQ_edge_list)

        with open(qua_file, "r") as fin:
            lines = fin.readlines()
            QA_edge_list = []
            for line in lines:
                unit = line.strip().split()
                QA_edge_list.append(["Q_" + unit[0],
                                     "A_" + unit[1]])
            G.add_edges_from(QA_edge_list)


    def generate_walk(self):
        """ Generate Random Walk

        Generating random walk from the Tripatite graph

        Args:

        """
        G = self.G
        num_walks, walk_len = self._num_walks, self._walk_length
        rand = random.Random(0)

        if not G.number_of_edges() or not G.number_of_nodes():
            sys.exit("Graph not initialized!")

        walks = []
        nodes = list(G.nodes)

        for cnt in range(num_walks):
            rand.shuffle(nodes)
            for node in nodes:
                walks.append(random_walk(start=node, alpha=alpha))

        return walks


    def random_walk(self, start=None):
        """Single Walk Generator

        Generating a single random walk that follows a meta path of
        `... R - Q - A - Q - R - Q - A - Q - R - Q ...`

        Args:
            start - starting node

        """

        return walks


