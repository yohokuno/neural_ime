#!/usr/bin/env python3
import argparse
import json
import os
import time

from utility import load_train_data
from rnn_trainer import RNNTrainer
from train import train_epoch
from rnn_predictor import RNNPredictor
from decode import load_dictionary, decode
from evaluate import evaluate


def decode_all(rnn_predictor, valid_source_data, dictionary, beam_size, viterbi_size):
    start_time = time.time()
    system = []
    for i, source in enumerate(valid_source_data):
        start_time_sentence = time.time()
        top_result, _, _, _ = decode(source, dictionary, rnn_predictor, beam_size, viterbi_size)
        decode_time_sentence = time.time() - start_time_sentence
        print('decoding sentence: {} time: {:.2f}'.format(i, decode_time_sentence), end='\r')
        system.append(top_result)

    decode_time = time.time() - start_time
    return system, decode_time


def train(rnn_trainer, rnn_predictor, train_data, valid_target_data, valid_source_data, dictionary,
          epoch_size, model_directory, beam_size, viterbi_size):
    start_time = time.time()
    log_path = os.path.join(model_directory, 'log.txt')
    log_file = open(log_path, 'w')
    best_epoch = None
    best_metrics = None

    for epoch in range(epoch_size):
        # Train one epoch and save the model
        train_epoch(rnn_trainer, train_data, model_directory, epoch)

        # Decode all sentences
        rnn_predictor.restore_from_directory(model_directory)
        system, decode_time = decode_all(rnn_predictor, valid_source_data, dictionary, beam_size, viterbi_size)

        # Evaluate results
        metrics = evaluate(system, valid_target_data)

        # Print metrics
        log_text = 'decoding precision: {:.2f} recall: {:.2f} f-score: {:.2f} accuracy: {:.2f}\n'.format(*metrics)
        log_text += 'decoding total time: {:.2f} average time: {:.2f}'.format(decode_time, decode_time / len(system))
        print(log_text)
        print(log_text, file=log_file)

        # Write decoded results to file
        decode_path = os.path.join(model_directory, 'decode-{}.txt'.format(epoch))
        with open(decode_path, 'w') as file:
            file.write('\n'.join(system))

        # Update best epoch
        if not best_epoch or best_metrics[2] < metrics[2]:
            best_epoch = epoch
            best_metrics = metrics

    total_time = time.time() - start_time
    print('best epoch:', best_epoch)
    print('best epoch metrics: precision: {:.2f} recall: {:.2f} f-score: {:.2f} accuracy: {:.2f}'.format(*best_metrics))
    print('total experiment time:', total_time)
    print()
    return best_metrics, best_epoch


def experiment(settings):
    # Print settings
    print(json.dumps(vars(settings), indent=4))

    # Load and preprocess training data
    train_data = load_train_data(settings)
    print('number of batches:', len(train_data))

    # Load validation data
    valid_target_data = [line.rstrip('\n') for line in open(settings.valid_target_file)]
    valid_source_data = [line.rstrip('\n') for line in open(settings.valid_source_file)]

    # Load dictionary for decoding
    dictionary = load_dictionary(settings.model_directory)

    # Create RNN model for training
    rnn_trainer = RNNTrainer(settings.batch_size, settings.sentence_size, settings.vocabulary_size, settings.hidden_size,
                             settings.layer_size, settings.cell_type, settings.optimizer_type, settings.clip_norm,
                             settings.keep_prob, settings.max_keep)

    # Create RNN model for prediction
    rnn_predictor = RNNPredictor(settings.vocabulary_size, settings.hidden_size, settings.layer_size, settings.cell_type)

    # Run experiment
    result = train(rnn_trainer, rnn_predictor, train_data, valid_target_data, valid_source_data, dictionary,
                 settings.epoch_size, settings.model_directory, settings.beam_size, settings.viterbi_size)

    rnn_trainer.close()
    rnn_predictor.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_file')
    parser.add_argument('valid_target_file')
    parser.add_argument('valid_source_file')
    parser.add_argument('model_directory')
    parser.add_argument('--sentence_size', type=int, default=30)
    parser.add_argument('--vocabulary_size', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--hidden_size', type=int, default=400)
    parser.add_argument('--layer_size', type=int, default=1)
    parser.add_argument('--epoch_size', type=int, default=10)
    parser.add_argument('--clip_norm', type=float, default=5)
    parser.add_argument('--keep_prob', type=float, default=0.5)
    parser.add_argument('--cell_type', default='gru')
    parser.add_argument('--optimizer_type', default='adam')
    parser.add_argument('--max_keep', type=int, default=0)
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--viterbi_size', type=int, default=1)
    args = parser.parse_args()
    experiment(args)

if __name__ == '__main__':
    main()
