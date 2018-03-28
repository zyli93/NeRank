from generate_walk import MetaPathGenerator
from preprocessing import preprocess

import os, sys
from optparse import OptionParser


def runPDER(options):

    # Check validity of parameters
    if type(options.preprocess) is not bool:
        print("Invalid -p value. Should be \"True\" or \"False\"",
              file=sys.stderr)
        sys.exit()

    if options.preprocess:
        preprocess(options.dataset)


    # preprocessing
    mp_generator = MetaPathGenerator(length=options.length,
                                     num_walks=options.size,
                                     dataset=options.dataset)
    walks = mp_generator.generate_metapaths(patterns=options.meta_paths.split(" "),
                                            alpha=options.alpha)
    mp_generator.write_metapaths(walks)




if __name__ == '__main__':
    """Generating random walks and output to file

    Args: 
        -d, --dataset
        -l, --length 
        -s, --size
        -a, --alpha
        -m, --meta-paths
        -p, --preprocess
        -w, --window-size

    Returns:
        Write generated meta-path

    """

    parser = OptionParser()
    parser.add_option("-d", "--dataset", type="string",
                      dest="dataset", default="3dprinting",
                      help="The dataset to work on.")
    parser.add_option("-l", "--length", type="int",
                      dest="length", default=15,
                      help="The length of the random walk to be generated.")
    parser.add_option("-s", "--size", type="int",
                      dest="size", default=2,
                      help="The number of times of each node to be iterated.")
    parser.add_option("-a", "--alpha", type="float",
                      dest="alpha", default=0.0,
                      help="The probability of restarting in meta-path generating")
    parser.add_option("-m", "--meta-paths", type="string",
                      dest="meta_paths",
                      help="The target meta-paths used to generate the data file, "
                           "split by space, enclose by \"\".")
    parser.add_option("-p", "--preprocess", default=False,
                      dest="preprocess", action="store_true",
                      help="Adding it to indicate doing preprocessing.")
    parser.add_option("-w", "--window-size", type="int",
                      dest="window_size", default=5,
                      help="The window size of the meta-path model.")

    (options, args) = parser.parse_args()
    runPDER(options)
