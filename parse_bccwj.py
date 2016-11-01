#!/usr/bin/env python3
import argparse
import glob
import xml.etree.ElementTree


def parse_sentence(sentence):
    for element in sentence.iter('SUW'):
        target = ''.join(element.itertext())
        target = target.replace('\n', '')
        if '　' in target:
            # Ignore full width space
            continue

        if 'kana' in element.attrib:
            source = element.attrib['kana']
        else:
            source = element.attrib['formBase']
        if source == '':
            source = target

        yield target, source


def parse_file(filename):
    tree = xml.etree.ElementTree.parse(filename)

    for sentence in tree.iter('sentence'):
        sentence = list(parse_sentence(sentence))
        if sentence:
            yield sentence


def parse_pathname(pathname):
    for filename in glob.glob(pathname):
        corpus = parse_file(filename)
        for sentence in corpus:
            yield sentence


def katakana_to_hiragana(string):
    result = ''
    for character in string:
        code = ord(character)
        # if 0x30A1 <= code < = 0x30F6:
        if ord('ァ') <= code <= ord('ヶ'):
            # result += chr(code + 0x3041 - 0x30A1)
            result += chr(code - ord('ァ') + ord('ぁ'))
        else:
            result += character
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pathname')
    args = parser.parse_args()

    sentences = parse_pathname(args.pathname)
    for sentence in sentences:
        line = ' '.join('{}/{}'.format(target, katakana_to_hiragana(source)) for target, source in sentence)
        print(line)


if __name__ == '__main__':
    main()
