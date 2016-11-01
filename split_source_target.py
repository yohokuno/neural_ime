#!/usr/bin/env python3
import argparse


def parse_file(file):
    for line in file:
        line = line.rstrip('\n')
        words = line.split(' ')
        yield [word.split('/', 1) for word in words]


def split_source_target(sentences):
    target = ''
    source = ''
    for sentence in sentences:
        for target_word, source_word in sentence:
            target += target_word
            source += source_word
        target += '\n'
        source += '\n'
    return target, source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=argparse.FileType('r'))
    parser.add_argument('target_file', type=argparse.FileType('w'))
    parser.add_argument('source_file', type=argparse.FileType('w'))
    args = parser.parse_args()

    sentences = parse_file(args.input_file)
    target, source = split_source_target(sentences)
    args.target_file.write(target)
    args.source_file.write(source)

if __name__ == '__main__':
    main()
