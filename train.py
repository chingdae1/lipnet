import argparse
from solver import Solver
import random, string


def randomword(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


def main():
    parser = argparse.ArgumentParser()
    # Training
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--decay_after', type=int, default=3)
    parser.add_argument('--decay_every', type=int, default=2)
    parser.add_argument('--decay_rate', type=float, default=0.1)

    # Dataset
    parser.add_argument('--num_of_frame', type=int, default=75)
    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--vocab_size', type=int, default=27)

    # Model
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='./trained_models/model_{}'.format(randomword(7)))

    config = parser.parse_args()
    solver = Solver(config)
    solver.fit()

if __name__ == '__main__':
    main()
