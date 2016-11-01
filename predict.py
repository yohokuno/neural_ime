#!/usr/bin/env python3
import argparse
import os
import collections
import numpy as np

from decode import load_settings
from rnn_predictor import RNNPredictor


def load_dictionary(model_directory):
    vocabulary_path = os.path.join(model_directory, 'vocabulary.txt')
    dictionary = collections.defaultdict(int)
    for word_id, line in enumerate(open(vocabulary_path)):
        word = line.rstrip('\n')
        dictionary[word] = word_id
    vocabulary = [line.rstrip('\n') for line in open(vocabulary_path)]
    return dictionary, vocabulary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_file', type=argparse.FileType('r'))
    parser.add_argument('model_directory')
    parser.add_argument('--model_file')
    args = parser.parse_args()

    # Load settings and vocabulary
    settings = load_settings(args.model_directory)
    dictionary, vocabulary = load_dictionary(args.model_directory)

    # Create model and load parameters
    rnn_predictor = RNNPredictor(settings.vocabulary_size, settings.hidden_size, settings.layer_size, settings.cell_type)
    if args.model_file:
        rnn_predictor.restore_from_file(args.model_file)
    else:
        rnn_predictor.restore_from_directory(args.model_directory)

    all_predictions = 0
    correct_predictions = 0

    for line in args.test_file:
        line = line.rstrip('\n')
        words = line.split(' ')
        words = ['_BOS/_BOS'] + words + ['_EOS/_EOS']
        state = None

        for i in range(len(words) - 2):
            word_id = dictionary[words[i]]
            predictions, state = rnn_predictor.predict([word_id], state)
            prediction = vocabulary[np.argmin(predictions[0])]

            if prediction == words[i + 1]:
                correct_predictions += 1
            all_predictions += 1

        print(correct_predictions / all_predictions, end='\r')

    print(correct_predictions / all_predictions)

if __name__ == '__main__':
    main()
