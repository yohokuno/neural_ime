#!/usr/bin/env python3
import argparse


def get_common_length(left, right):
    # Compute length of the longest common sub-sequence of two strings
    table = [[0 for _ in range(len(right) + 1)] for _ in range(len(left) + 1)]

    for i in range(1, len(left) + 1):
        for j in range(1, len(right) + 1):
            if left[i - 1] == right[j - 1]:
                table[i][j] = table[i-1][j-1] + 1
            else:
                table[i][j] = max(table[i-1][j], table[i][j-1])
    return table[-1][-1]


def evaluate(system, reference):
    # extract statistics
    common_length = sum(get_common_length(r, s) for r, s in zip(reference, system))
    reference_length = len(''.join(reference))
    system_length = len(''.join(system))
    sentence_match = sum(r == s for r, s in zip(reference, system))

    # calculate metrics
    if system_length > 0:
        precision = 100. * common_length / system_length
    else:
        precision = 0.
    recall = 100. * common_length / reference_length
    fscore = 200. * common_length / (reference_length + system_length)
    accuracy = 100. * sentence_match / len(reference)

    # return metrics
    return precision, recall, fscore, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('system', type=argparse.FileType('r'))
    parser.add_argument('reference', type=argparse.FileType('r'))
    args = parser.parse_args()

    # load data
    system = [line.rstrip('\n') for line in args.system]
    reference = [line.rstrip('\n') for line in args.reference]
    reference = reference[:len(system)]

    # calculate metrics
    metrics = evaluate(system, reference)

    # print metrics
    print('precision: {:.2f} recall: {:.2f} f-score: {:.2f} accuracy: {:.2f}'.format(*metrics))


if __name__ == '__main__':
    main()
