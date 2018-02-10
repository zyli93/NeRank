"""Random walk generator

Generating random walks on our Uq, Ua, and Q network using NetworkX.

Author: Zeyu Li zyli@cs.ucla.edu

"""

import os, sys, argparse
import networkx as nx


class RandomWalkGenerator:
    """RandomWalkGenerator

    Args:
        length - the length of random walks to be generated
        size   - the number of random walks to be generated


    """

    def __init__(self, length=100, size=10000):
        self._length = length
        self._size = size
        self.G = nx.Graph()

    def initialize(self, uqq_file, qua_file):
        """ Initialize Graph

        Initialize graph with Uq-Q pairs and Q-Ua pairs.

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




