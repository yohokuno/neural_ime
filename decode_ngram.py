#!/usr/bin/env python3
import argparse
import sys
import heapq
import operator
import math
from collections import defaultdict


def parse_ngram(ngram):
    for word in ngram.split(' '):
        if word == '<s>':
            yield '_BOS', '_BOS'
        elif word == '</s>':
            yield '_EOS', '_EOS'
        else:
            yield tuple(word.split('/', 1))


def parse_srilm(file):
    ngrams = {}
    for line in file:
        line = line.rstrip('\n')
        fields = line.split('\t', 2)

        if len(fields) < 2:
            continue

        if len(fields) == 2:
            logprob, ngram = fields
            backoff = None
        elif len(fields) == 3:
            logprob, ngram, backoff = fields
            backoff = -math.log(10 ** float(backoff))
        cost = -math.log(10 ** float(logprob))
        ngram = tuple(parse_ngram(ngram))
        ngrams[ngram] = (cost, backoff)
    return ngrams


def create_dictionary(ngrams):
    dictionary = defaultdict(list)
    for ngram in ngrams.keys():
        if len(ngram) == 1:
            target, source = ngram[0]
            dictionary[source].append(target)
    return dictionary


def create_lattice(input_, dictionary):
    lattice = [[[] for _ in range(len(input_) + 1)] for _ in range(len(input_) + 2)]

    for i in range(1, len(input_) + 1):
        for j in range(i):
            source = input_[j:i]
            if source in dictionary:
                for target in dictionary[source]:
                    lattice[i][j].append((target, source))
            elif len(source) == 1:
                lattice[i][j].append((source, source))

    lattice[-1][-1].append(('_EOS', '_EOS'))
    return lattice


def initialize_queues(lattice):
    # A hypothesis is tuple of (cost, history)
    queues = [[] for _ in range(len(lattice))]
    bos_hypothesis = (0.0, [('_BOS', '_BOS')])
    queues[0].append(bos_hypothesis)
    return queues


def get_ngram_cost(ngrams, history):
    if type(history) is list:
        history = tuple(history)
    if history in ngrams:
        cost, _ = ngrams[history]
        return cost

    if len(history) == 1:
        return 100.0

    return get_ngram_cost(ngrams, history[1:])


def search(lattice, ngrams, queues, beam_size, viterbi_size):
    for i in range(len(lattice)):
        for j in range(len(lattice[i])):
            for target, source in lattice[i][j]:

                word_queue = []
                for previous_cost, previous_history in queues[j]:
                    history = previous_history + [(target, source)]
                    cost = previous_cost + get_ngram_cost(ngrams, tuple(history[-3:]))
                    hypothesis = (cost, history)
                    word_queue.append(hypothesis)

                # prune word_queue to viterbi size
                if viterbi_size > 0:
                    word_queue = heapq.nsmallest(viterbi_size, word_queue, key=operator.itemgetter(0))

                queues[i] += word_queue

        # prune queues[i] to beam size
        if beam_size > 0:
            queues[i] = heapq.nsmallest(beam_size, queues[i], key=operator.itemgetter(0))
    return queues


def decode(input_, dictionary, ngrams, beam_size, viterbi_size):
    lattice = create_lattice(input_, dictionary)
    queues = initialize_queues(lattice)
    queue = search(lattice, ngrams, queues, beam_size, viterbi_size)

    candidates = []
    for cost, history in queue[-1]:
        result = ''.join(target for target, source in history[1:-1])
        candidates.append((result, cost))

    top_result = candidates[0][0]
    return top_result, candidates, lattice, queues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ngram_file')
    parser.add_argument('--input_file', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument('--output_file', type=argparse.FileType('w'), default=sys.stdout)
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--viterbi_size', type=int, default=1)
    parser.add_argument('--print_nbest', action='store_true')
    parser.add_argument('--print_queue', action='store_true')
    parser.add_argument('--print_lattice', action='store_true')
    args = parser.parse_args()

    ngrams = parse_srilm(open(args.ngram_file))
    dictionary = create_dictionary(ngrams)

    for line in args.input_file:
        line = line.rstrip('\n')
        result, candidates, lattice, queues = decode(line, dictionary, ngrams, args.beam_size, args.viterbi_size)

        # Print decoded results
        if not args.print_nbest:
            print(result, file=args.output_file)
        else:
            for string, cost in candidates:
                print(string, cost, file=args.output_file)

        # Print lattice for debug
        if args.print_lattice:
            for i in range(len(lattice)):
                for j in range(len(lattice[i])):
                    print('i = {}, j = {}'.format(i, j), file=args.output_file)
                    for target, source in lattice[i][j]:
                        print(target, source, file=args.output_file)

        # Print queues for debug
        if args.print_queue:
            for i, queue in enumerate(queues):
                print('queue', i, file=args.output_file)
                for cost, history in queue:
                    print(cost, history, file=args.output_file)

if __name__ == '__main__':
    main()
