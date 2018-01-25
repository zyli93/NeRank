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









