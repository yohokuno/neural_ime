from collections import Counter
import os
import json


def parse_file(file):
    for line in file:
        line = line.rstrip('\n')
        sentence = line.split(' ')
        yield sentence


# TODO: current method does not allow the model to learn boundary beyond bigram.
def adjust_size(sentences, sentence_size):
    # Increment sentence size for shifting output later
    sentence_size_plus = sentence_size + 1

    for sentence in sentences:
        # Insert BOS = Beginning Of Sentence
        sentence.insert(0, '_BOS/_BOS')

        # Split long sentence allowing overlap of 1 word
        while len(sentence) >= sentence_size_plus:
            yield sentence[:sentence_size_plus]
            sentence = sentence[sentence_size:]

        # Do not yield EOS-only sentence
        if sentence:
            # Insert EOS = End Of Sentence
            sentence.append('_EOS/_EOS')

            if len(sentence) < sentence_size_plus:
                # Padding sentence to make its size sentence_size_plus
                sentence += ['_PAD/_PAD'] * (sentence_size_plus - len(sentence))
            yield sentence


def create_vocabulary(sentences, vocabulary_size):
    # Create list of words indexed by word ID
    counter = Counter(word for words in sentences for word in words)
    most_common = counter.most_common(vocabulary_size - 1)
    vocabulary = [word for word, count in most_common]
    vocabulary.insert(0, '_UNK/_UNK')
    return vocabulary


def convert_to_ids(sentences, vocabulary):
    dictionary = dict((word, word_id) for word_id, word in enumerate(vocabulary))

    for sentence in sentences:
        word_ids = []

        for word in sentence:
            if word in dictionary:
                word_id = dictionary[word]
            else:
                word_id = dictionary['_UNK/_UNK']
            word_ids.append(word_id)

        yield word_ids


# TODO: current batching ignores sentences that does't fit into last batch.
def create_batches(sentences, batch_size):
    all_batches = int(len(sentences) / batch_size)

    for i in range(all_batches):
        batch_sentences = sentences[i * batch_size:(i + 1) * batch_size]
        batch_input = []
        batch_output = []

        for sentence in batch_sentences:
            # Shift sentence by 1 time step
            input_ = sentence[:-1]
            output_ = sentence[1:]

            batch_input.append(input_)
            batch_output.append(output_)

        yield batch_input, batch_output


def save_metadata(args, vocabulary):
    # Create directory if not exists
    if not os.path.exists(args.model_directory):
        os.makedirs(args.model_directory)

    # Save settings
    settings_path = os.path.join(args.model_directory, 'settings.json')
    with open(settings_path, 'w') as settings_file:
        json.dump(vars(args), settings_file, indent=4)

    # Save vocabulary
    vocabulary_path = os.path.join(args.model_directory, 'vocabulary.txt')
    with open(vocabulary_path, 'w') as vocabulary_file:
        vocabulary_file.write('\n'.join(vocabulary))


def load_train_data(args):
    sentences = parse_file(open(args.train_file))
    sentences = list(adjust_size(sentences, args.sentence_size))
    vocabulary = create_vocabulary(sentences, args.vocabulary_size)
    sentences = list(convert_to_ids(sentences, vocabulary))
    train_data = list(create_batches(sentences, args.batch_size))
    save_metadata(args, vocabulary)
    return train_data
