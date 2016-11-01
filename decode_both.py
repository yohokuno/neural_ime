#!/usr/bin/env python3
import argparse
import sys
import heapq
import operator
import math

from rnn_predictor import RNNPredictor
from decode import load_settings, load_dictionary
from decode_ngram import parse_srilm, get_ngram_cost


def create_lattice(input_, dictionary):
    lattice = [[[] for _ in range(len(input_) + 1)] for _ in range(len(input_) + 2)]
    _, unk_id = dictionary['_UNK'][0]

    for i in range(1, len(input_) + 1):
        for j in range(i):
            source = input_[j:i]
            if source in dictionary:
                for target, word_id in dictionary[source]:
                    lattice[i][j].append((target, source, word_id))
            elif len(source) == 1:
                # Create _UNK node with verbatim target when single character key is not found in the dictionary.
                lattice[i][j].append((source, source, unk_id))

    _, eos_id = dictionary['_EOS'][0]
    lattice[-1][-1].append(('_EOS', '_EOS', eos_id))
    return lattice


def initialize_queues(lattice, rnn_predictor, dictionary):
    # A hypothesis is tuple of (cost, history, state, prediction)
    _, bos_id = dictionary['_BOS'][0]
    bos_predictions, bos_states = rnn_predictor.predict([bos_id])
    bos_hypothesis = (0.0, [('_EOS', '_EOS')], bos_states[0], bos_predictions[0])
    queues = [[] for _ in range(len(lattice))]
    queues[0].append(bos_hypothesis)
    return queues


def interpolate(rnn_cost, ngram_cost):
    # Linear interpolation needs to be done in probability space, not log probability space
    return -math.log((math.exp(-rnn_cost) + math.exp(-ngram_cost)) / 2.0)


def search(lattice, queues, rnn_predictor, ngrams, beam_size, viterbi_size):
    # Breadth first search with beam pruning and viterbi-like pruning
    for i in range(len(lattice)):
        queue = []

        # create hypotheses without predicting next word
        for j in range(len(lattice[i])):
            for target, source, word_id in lattice[i][j]:

                word_queue = []
                for previous_cost, previous_history, previous_state, previous_prediction in queues[j]:
                    history = previous_history + [(target, source)]
                    cost = previous_cost + interpolate(previous_prediction[word_id], get_ngram_cost(ngrams, history))
                    # Temporal hypothesis is tuple of (cost, history, word_id, previous_state)
                    # Lazy prediction replaces word_id and previous_state to state and prediction
                    hypothesis = (cost, history, word_id, previous_state)
                    word_queue.append(hypothesis)

                # prune word_queue to viterbi size
                if viterbi_size > 0:
                    word_queue = heapq.nsmallest(viterbi_size, word_queue, key=operator.itemgetter(0))

                queue += word_queue

        # prune queue to beam size
        if beam_size > 0:
            queue = heapq.nsmallest(beam_size, queue, key=operator.itemgetter(0))

        # predict next word and state before continue
        for cost, history, word_id, previous_state in queue:
            predictions, states = rnn_predictor.predict([word_id], [previous_state])
            hypothesis = (cost, history, states[0], predictions[0])
            queues[i].append(hypothesis)

    return queues


def decode(source, dictionary, rnn_predictor, ngrams, beam_size, viterbi_size):
    lattice = create_lattice(source, dictionary)
    queues = initialize_queues(lattice, rnn_predictor, dictionary)
    queues = search(lattice, queues, rnn_predictor, ngrams, beam_size, viterbi_size)

    candidates = []
    for cost, history, _, _ in queues[-1]:
        result = ''.join(target for target, source in history[1:-1])
        candidates.append((result, cost))

    top_result = candidates[0][0]
    return top_result, candidates, lattice, queues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_directory')
    parser.add_argument('ngram_file')
    parser.add_argument('--model_file')
    parser.add_argument('--input_file', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument('--output_file', type=argparse.FileType('w'), default=sys.stdout)
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--viterbi_size', type=int, default=1)
    parser.add_argument('--print_nbest', action='store_true')
    parser.add_argument('--print_queue', action='store_true')
    parser.add_argument('--print_lattice', action='store_true')
    args = parser.parse_args()

    # Load settings and vocabulary
    settings = load_settings(args.model_directory)
    dictionary = load_dictionary(args.model_directory)

    # Create model and load parameters
    rnn_predictor = RNNPredictor(settings.vocabulary_size, settings.hidden_size, settings.layer_size, settings.cell_type)
    if args.model_file:
        rnn_predictor.restore_from_file(args.model_file)
    else:
        rnn_predictor.restore_from_directory(args.model_directory)

    # Load ngram file in SRILM format
    ngrams = parse_srilm(open(args.ngram_file))

    # Iterate input file line by line
    for line in args.input_file:
        line = line.rstrip('\n')

        # Decode - this might take some time
        result, candidates, lattice, queues = decode(line, dictionary, rnn_predictor, ngrams, args.beam_size, args.viterbi_size)

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
                    for target, source, word_id in lattice[i][j]:
                        print(target, source, word_id, file=args.output_file)

        # Print queues for debug
        if args.print_queue:
            for i, queue in enumerate(queues):
                print('queue', i, file=args.output_file)
                for cost, history, state, prediction in queue:
                    print(cost, history, file=args.output_file)


if __name__ == '__main__':
    main()
