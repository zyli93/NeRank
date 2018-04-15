from generate_walk import MetaPathGenerator
from preprocessing import preprocess
from pder2 import PDER

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
    if options.gen_mp:
        mp_generator = MetaPathGenerator(
            length=options.length,
            num_walks=options.coverage,
            dataset=options.dataset)
        walks = mp_generator.generate_metapaths(
            patterns=options.meta_paths.split(" "),
            alpha=options.alpha)
        mp_generator.write_metapaths(walks)

    # init data_loader
    pder_model = PDER(
        dataset=options.dataset,
        embedding_dim=options.embedding_dim,
        epoch_num=options.epoch_num,
        batch_size=options.batch_size,
        window_size=options.window_size,
        neg_sample_ratio=options.neg_ratio,
        lstm_layers=options.lstm_layers,
        include_content=options.include_content
    )

    pder_model.train()
    pder_model.test()



if __name__ == '__main__':
    """Generating random walks and output to file

    Args:
        -d, --dataset (str)
        -l, --length (int)
        -c, --coverage (int)
        -a, --alpha (int)
        -m, --meta-paths (str, split by " ")
        -p, --preprocess (bool)
        -w, --window-size (size)
        -g, --gen-metapaths (bool)
        -n, --neg-ratio (float)
        -e, --embedding-dim (int)
        -y, --lstm-layers (int)
        -p, --epoch-number (int)
        -b, --batch-size (int)
        -u, --include-content (bool)

    Returns:
        do everything

    """

    parser = OptionParser()
    parser.add_option("-d", "--dataset", type="string",
                      dest="dataset", default="3dprinting",
                      help="The dataset to work on.")

    parser.add_option("-l", "--length", type="int",
                      dest="length", default=15,
                      help="The length of the random walk to be generated.")

    parser.add_option("-c", "--coverage", type="int",
                      dest="coverage", default=2,
                      help="The number of times each node to be covered.")

    parser.add_option("-a", "--alpha", type="float",
                      dest="alpha", default=0.0,
                      help="The probability of restarting in meta-path generating")

    parser.add_option("-m", "--metapaths", type="string",
                      dest="meta_paths",
                      help="The target meta-paths used to generate the data file, "
                           "split by space, enclose by \"\".")

    parser.add_option("-p", "--preprocess", default=False,
                      dest="preprocess", action="store_true",
                      help="Adding it to indicate doing preprocessing.")

    parser.add_option("-w", "--window-size", type="int",
                      dest="window_size", default=5,
                      help="The window size of the meta-path model.")

    parser.add_option("-g", "--gen-metapaths", default=False,
                      dest="gen_mp", action="store_true",
                      help="Decide whether to generate new metapaths.")

    parser.add_option("-n", "--neg-ratio", type="float",
                      dest="neg_ratio", default=1.2,
                      help="The ratio of negative samples.")

    parser.add_option("-e", "--embedding-dim", type="int",
                      dest="embedding_dim", default=300,
                      help="The embedding dimension of the model.")

    parser.add_option("-y", "--lstm-layers", type="int",
                      dest="lstm_layers", default=3,
                      help="The number of layers of the LSTM model.")

    parser.add_option("-o", "--epoch-number", type="int",
                      dest="epoch_num", default=1000,
                      help="The epoch number of the training set.")

    parser.add_option("-b", "--batch-size", type="int",
                      dest="batch_size", default=10,
                      help="The number of meta-paths fed into the model each batch")

    parser.add_option("-u", "--include-content", default=False,
                      dest="include_content", action="store_true",
                      help="Whether to include content in the text embedding")

    parser.add_option("-r", "--learning-rate", type="float",
                      dest="learning_rate", default=0.01,
                      help="The learning rate.")


    (options, args) = parser.parse_args()

    runPDER(options)
