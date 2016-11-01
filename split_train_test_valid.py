#!/usr/bin/env python3
import argparse
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=argparse.FileType('r'))
    parser.add_argument('train_file', type=argparse.FileType('w'))
    parser.add_argument('test_file', type=argparse.FileType('w'))
    parser.add_argument('valid_file', type=argparse.FileType('w'))
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--valid_size', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    input_data = args.input_file.readlines()

    random.seed(args.seed)
    random.shuffle(input_data)

    test_index = int(len(input_data) * args.test_size)
    test_data = input_data[:test_index]

    valid_index = test_index + int(len(input_data) * args.valid_size)
    valid_data = input_data[test_index:valid_index]
    train_data = input_data[valid_index:]

    args.train_file.writelines(train_data)
    args.test_file.writelines(test_data)
    args.valid_file.writelines(valid_data)

if __name__ == '__main__':
    main()
