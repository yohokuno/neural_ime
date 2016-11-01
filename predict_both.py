#!/usr/bin/env python3
import argparse
import numpy as np

from decode import load_settings
from decode_ngram import get_ngram_cost, parse_srilm
from predict import load_dictionary
from rnn_predictor import RNNPredictor


def match_predictions(rnn_predictor, dictionary, vocabulary, ngrams, words):
    state = None

    for i in range(len(words) - 2):
        # RNN prediction
        word_id = dictionary[words[i]]
        predictions, state = rnn_predictor.predict([word_id], state)
        rnn_prediction = predictions[0]

        # N-gram prediction
        context = words[max(i - 3, 0):i + 1]
        ngram_prediction = np.zeros(len(rnn_prediction))
        for word in list(dictionary.values()):
            history = tuple(context + [word])
            probability = get_ngram_cost(ngrams, history)
            word_id = dictionary[word]
            ngram_prediction[word_id] = probability

        interpolation = -np.log((np.exp(-rnn_prediction) + np.exp(-ngram_prediction)) / 2.0)
        prediction = vocabulary[np.argmin(interpolation)]
        yield prediction == words[i + 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_file', type=argparse.FileType('r'))
    parser.add_argument('model_directory')
    parser.add_argument('ngram_file', type=argparse.FileType('r'))
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

    # Load N-gram model
    ngrams = parse_srilm(args.ngram_file)

    all_predictions = 0
    correct_predictions = 0

    for line in args.test_file:
        line = line.rstrip('\n')
        words = line.split(' ')
        words = ['_BOS/_BOS'] + words + ['_EOS/_EOS']

        result = list(match_predictions(rnn_predictor, dictionary, vocabulary, ngrams, words))
        all_predictions += len(result)
        correct_predictions += sum(result)
        print(correct_predictions / all_predictions, end='\r')

    print(correct_predictions / all_predictions)


if __name__ == '__main__':
    main()
