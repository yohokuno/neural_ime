#!/usr/bin/env python3
import argparse
import json
import os
import time
import itertools
from experiment import experiment


def grid_search(args):
    start_time = time.time()

    results = []

    # split args to default settings and hyperparameters
    default_settings = {}
    search_settings = {}
    for name, value in vars(args).items():
        if type(value) == list and len(value) == 1:
            value = value[0]
        if type(value) != list:
            default_settings[name] = value
        else:
            search_settings[name] = value

    # Search hyperparameters
    for values in itertools.product(*search_settings.values()):
        # Merge default and search settings
        hyperparameters = dict(zip(search_settings.keys(), values))
        merged_settings = {**default_settings, **hyperparameters}
        settings = argparse.Namespace(**merged_settings)

        # Set directory path
        directory_name = '-'.join(name + str(value) for name, value in hyperparameters.items())
        settings.model_directory = os.path.join(settings.model_directory, directory_name)

        # Run experiment
        metrics, epoch = experiment(settings)
        result = (hyperparameters, metrics, epoch)
        results.append(result)

    # print best experiment result
    hyperparameters, metrics, epoch = max(results, key=lambda x: x[1][2])
    print('best experiment settings in epoch', epoch)
    print(json.dumps(hyperparameters, indent=4))
    print('best experiment metrics: precision: {:.2f} recall: {:.2f} f-score: {:.2f} accuracy: {:.2f}'.format(*metrics))

    # save all results
    results_path = os.path.join(args.model_directory, 'results.json')
    json.dump(results, open(results_path, 'w'), indent=4)

    # Print total time
    total_time = time.time() - start_time
    print('total grid search time:', total_time)


def main():
    parser = argparse.ArgumentParser()
    # mandatory parameters
    parser.add_argument('train_file')
    parser.add_argument('valid_target_file')
    parser.add_argument('valid_source_file')
    parser.add_argument('model_directory')
    # grid search parameters
    parser.add_argument('--hidden_size', type=int, default=400, nargs='*')
    parser.add_argument('--layer_size', type=int, default=1, nargs='*')
    parser.add_argument('--keep_prob', type=float, default=0.5, nargs='*')
    parser.add_argument('--cell_type', default='gru', nargs='*')
    parser.add_argument('--optimizer_type', default='adam', nargs='*')
    parser.add_argument('--vocabulary_size', type=int, default=50000, nargs='*')
    parser.add_argument('--sentence_size', type=int, default=30, nargs='*')
    parser.add_argument('--batch_size', type=int, default=50, nargs='*')
    parser.add_argument('--clip_norm', type=float, default=5, nargs='*')
    # optional parameters
    parser.add_argument('--epoch_size', type=int, default=10)
    parser.add_argument('--max_keep', type=int, default=0)
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--viterbi_size', type=int, default=1)
    args = parser.parse_args()

    # Run grid search. This might take long time to complete.
    grid_search(args)


if __name__ == '__main__':
    main()
