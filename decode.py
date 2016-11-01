#!/usr/bin/env python3
import argparse
import sys
import os
import json
import collections
import heapq
import operator
from rnn_predictor import RNNPredictor


def load_settings(model_directory):
    settings_path = os.path.join(model_directory, 'settings.json')
    settings = json.load(open(settings_path))
    return argparse.Namespace(**settings)


def load_dictionary(model_directory):
    vocabulary_path = os.path.join(model_directory, 'vocabulary.txt')
    vocabulary = []
    for line in open(vocabulary_path):
        line = line.rstrip('\n')
        target, source = line.split('/', 1)
        vocabulary.append((target, source))

    dictionary = collections.defaultdict(list)
    for i, (target, source) in enumerate(vocabulary):
        dictionary[source].append((target, i))

    return dictionary


def create_lattice(input_, dictionary):
    lattice = [[[] for _ in range(len(input_) + 1)] for _ in range(len(input_) + 2)]
    _, unk_id = dictionary['_UNK'][0]

    for i in range(1, len(input_) + 1):
        for j in range(i):
            key = input_[j:i]
            if key in dictionary:
                for target, word_id in dictionary[key]:
                    lattice[i][j].append((target, word_id))
            elif len(key) == 1:
                # Create _UNK node with verbatim target when single character key is not found in the dictionary.
                lattice[i][j].append((key, unk_id))

    _, eos_id = dictionary['_EOS'][0]
    lattice[-1][-1].append(('', eos_id))
    return lattice


def initialize_queues(lattice, rnn_predictor, dictionary):
    # Initialize priority queues for keeping hypotheses
    # A hypothesis is a tuple of (cost, string, state, prediction)
    # cost is total negative log probability
    # state.shape == [hidden_size * layer_size]
    # prediction.shape == [vocabulary_size]
    _, bos_id = dictionary['_BOS'][0]
    bos_predictions, bos_states = rnn_predictor.predict([bos_id])
    bos_hypothesis = (0.0, '', bos_states[0], bos_predictions[0])
    queues = [[] for _ in range(len(lattice))]
    queues[0].append(bos_hypothesis)
    return queues


def simple_search(lattice, queues, rnn_predictor, beam_size):
    # Simple but slow implementation of beam search
    for i in range(len(lattice)):
        for j in range(len(lattice[i])):
            for target, word_id in lattice[i][j]:
                for previous_cost, previous_string, previous_state, previous_prediction in queues[j]:
                    cost = previous_cost + previous_prediction[word_id]
                    string = previous_string + target
                    predictions, states = rnn_predictor.predict([word_id], [previous_state])
                    hypothesis = (cost, string, states[0], predictions[0])
                    queues[i].append(hypothesis)

        # prune queues[i] to beam size
        queues[i] = heapq.nsmallest(beam_size, queues[i], key=operator.itemgetter(0))
    return queues


def search(lattice, queues, rnn_predictor, beam_size, viterbi_size):
    # Breadth first search with beam pruning and viterbi-like pruning
    for i in range(len(lattice)):
        queue = []

        # create hypotheses without predicting next word
        for j in range(len(lattice[i])):
            for target, word_id in lattice[i][j]:

                word_queue = []
                for previous_cost, previous_string, previous_state, previous_prediction in queues[j]:
                    cost = previous_cost + previous_prediction[word_id]
                    string = previous_string + target
                    hypothesis = (cost, string, word_id, previous_state)
                    word_queue.append(hypothesis)

                # prune word_queue to viterbi size
                if viterbi_size > 0:
                    word_queue = heapq.nsmallest(viterbi_size, word_queue, key=operator.itemgetter(0))

                queue += word_queue

        # prune queue to beam size
        if beam_size > 0:
            queue = heapq.nsmallest(beam_size, queue, key=operator.itemgetter(0))

        # predict next word and state before continue
        for cost, string, word_id, previous_state in queue:
            predictions, states = rnn_predictor.predict([word_id], [previous_state])
            hypothesis = (cost, string, states[0], predictions[0])
            queues[i].append(hypothesis)

    return queues


def decode(source, dictionary, rnn_predictor, beam_size, viterbi_size):
    lattice = create_lattice(source, dictionary)
    queues = initialize_queues(lattice, rnn_predictor, dictionary)
    queues = search(lattice, queues, rnn_predictor, beam_size, viterbi_size)

    candidates = []
    for cost, string, _, _ in queues[-1]:
        candidates.append((string, cost))

    top_result = candidates[0][0]
    return top_result, candidates, lattice, queues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_directory')
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

    # Iterate input file line by line
    for line in args.input_file:
        line = line.rstrip('\n')

        # Decode - this might take ~10 seconds per line
        result, candidates, lattice, queues = decode(line, dictionary, rnn_predictor, args.beam_size, args.viterbi_size)

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
                    for target, word_id in lattice[i][j]:
                        print(target, word_id, file=args.output_file)

        # Print queues for debug
        if args.print_queue:
            for i, queue in enumerate(queues):
                print('queue', i, file=args.output_file)
                for cost, string, state, prediction in queue:
                    print(string, cost, file=args.output_file)

if __name__ == '__main__':
    main()
