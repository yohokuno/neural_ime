#!/usr/bin/env python3
import argparse
import json
import math
import os
import time

from utility import load_train_data
from rnn_trainer import RNNTrainer


def train_epoch(rnn_trainer, train_data, model_directory, epoch):
    total_loss = 0.0
    start_time = time.time()

    for batch, (input_, output_) in enumerate(train_data):
        start_time_batch = time.time()
        loss, gradient_norm = rnn_trainer.train(input_, output_)
        train_time_batch = time.time() - start_time_batch
        total_loss += loss
        perplexity = math.exp(loss)
        print('training batch: {:} perplexity: {:.2f} time: {:.2f}'.format(batch, perplexity, train_time_batch), end='\r')

    train_time = time.time() - start_time
    perplexity = math.exp(total_loss / len(train_data))

    log_text = 'training epoch: {} perplexity: {:.2f}               \n'.format(epoch, perplexity)
    log_text += 'training total time: {:.2f} average time: {:.2f}'.format(train_time, train_time / len(train_data))
    print(log_text)

    # Save model every epoch
    model_path = os.path.join(model_directory, 'model.ckpt')
    rnn_trainer.save(model_path, epoch)

    return perplexity, train_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_file')
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
    args = parser.parse_args()

    # Print settings
    print(json.dumps(vars(args), indent=4))

    # Load and preprocess training data
    train_data = load_train_data(args)
    print('number of batches:', len(train_data))

    # Create RNN model for training
    rnn_trainer = RNNTrainer(args.batch_size, args.sentence_size, args.vocabulary_size, args.hidden_size,
                             args.layer_size, args.cell_type, args.optimizer_type, args.clip_norm,
                             args.keep_prob, args.max_keep)

    start_time = time.time()

    for epoch in range(args.epoch_size):
        # Train one epoch and save the model
        train_epoch(rnn_trainer, train_data, args.model_directory, epoch)

    total_time = time.time() - start_time
    print('total training time:', total_time)
    print()

    rnn_trainer.close()

if __name__ == '__main__':
    main()
