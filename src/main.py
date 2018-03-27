from generate_walk import MetaPathGenerator

import os, sys
from optparse import OptionParser


if __name__ == '__main__':
    """Generating random walks and output to file

    Args: 
        -d, --dataset
        -l, --length 
        -s, --size
        -a, --alpha
        -m, --meta_paths

    Returns:
        Write generated meta-path

    """

    parser = OptionParser()
    parser.add_option("-d", "--dataset", type="string",
                      dest="dataset",
                      help="The dataset to work on.")
    parser.add_option("-l", "--length", type="int",
                      dest="length",
                      help="The length of the random walk to be generated.")
    parser.add_option("-s", "--size", type="int",
                      dest="size",
                      help="The number of times of each node to be iterated.")
    parser.add_option("-a", "--alpha", type="float",
                      dest="alpha",
                      help="The probability of restarting in meta-path generating")
    parser.add_option("-m", "--meta_paths", type="string",
                      dest="meta_paths",
                      help="The target meta-paths used to generate the data file, "
                           "split by space, enclose by \"\".")

    (options, args) = parser.parse_args()

    print("Generating Random Walks ...")
    mp_generator = MetaPathGenerator(length=options.)

    gw = MetaPathGenerator(length=15, num_walks=2, dataset="3dprinting")
    walks = gw.generate_metapaths(patterns=["AQRQA", "AQA"], alpha=0)
    gw.write_metapaths(walks)

