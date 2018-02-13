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

    def get_nodelist(self, type=None):
        """ Get specific type of nodelist in the graph

        Args:
            type - the entity type of the entity

        Return:
            nodelist - the list of node with `type`
        """
        G = self.G
        if not type:
            return list(G.nodes)
        return [node for node in list(G.nodes)
                if node[0] == type]

    def generate_walks(self, patterns, alpha):
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
        num_walks, walk_len = self._num_walks, self._walk_length
        rand = random.Random(0)

        if not G.number_of_edges() or not G.number_of_nodes():
            sys.exit("Graph not initialized!")

        walks = []
        nodes = list(G.nodes)

        for meta_pattern in patterns:
            for cnt in range(num_walks):
                nodes = self.get_nodelist()
                rand.shuffle(nodes)
                for node in self.get_nodelist("R"):
                    walks.append(
                        self.__meta_path_walk(
                            start=node, alpha=alpha,
                            pattern=meta_pattern,
                            rand=rand))
        return walks

    def __meta_path_walk(self, rand, start,
                         alpha=0.0, patterns="AQRQA", walk_len=50):
        """Single Walk Generator

        Generating a single random walk that follows a meta path of `pattern`

        Args:
            rand - an random object to generate random numbers
            start - starting node
            alpha - probability of restarts
            pattern - the pattern according to which to generate walks
            walk_len - the length of the generated walk

        Return:
            walk - the single walk generated

        """
        G = self.G





        return walks


