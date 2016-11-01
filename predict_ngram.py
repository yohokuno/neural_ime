#!/usr/bin/env python3
import argparse
import math
import collections


def parse_srilm(file):
    order = 0
    ngrams = collections.defaultdict(list)

    for line in file:
        line = line.rstrip('\n')
        fields = line.split('\t', 2)

        if len(fields) not in (2, 3):
            continue

        cost = -math.log(10 ** float(fields[0]))
        ngram = fields[1].split(' ')

        if len(ngram) > order:
            order = len(ngram)

        context = tuple(ngram[:-1])
        word = ngram[-1]
        ngrams[context].append((cost, word))

    return ngrams, order


def predict(ngrams, context):
    if context not in ngrams:
        return predict(ngrams, context[1:])

    cost, word = min(ngrams[context])
    return word


def match_predictions(ngrams, order, words):
    for i in range(1, len(words) - 1):
        word = words[i]
        context = tuple(words[max(i-order+1, 0):i])
        prediction = predict(ngrams, context)
        yield prediction == word


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_file', type=argparse.FileType('r'))
    parser.add_argument('ngram_file', type=argparse.FileType('r'))
    args = parser.parse_args()

    ngrams, order = parse_srilm(args.ngram_file)

    all_predictions = 0
    correct_predictions = 0

    for line in args.test_file:
        words = line.split(' ')
        words = ['<s>'] + words + ['</s>']
        result = list(match_predictions(ngrams, order, words))
        all_predictions += len(result)
        correct_predictions += sum(result)
        print(correct_predictions / all_predictions, end='\r')

    print(correct_predictions / all_predictions)


if __name__ == '__main__':
    main()
