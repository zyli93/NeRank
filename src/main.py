from generate_walk import RandomWalkGenerator

import argparse

a = RandomWalkGenerator(1,2)

if __name__ == '__main__':
    """Generating random walks and output to file

    Args: 
        -l, --length \t Length of the random walk to be generated 
        -n, --number \t Number of random walks to be generated 

    Returns:
        Write generated random   

    """
    opt_parser = argparse.ArgumentParser()
    opt_parser.add_argument("-l", "--length", type=int,
                            help="The length of the random walk to be generated.")
    opt_parser.add_argument("-s", "--size", type=int,
                            help="The number to total random walks to be generated.")
