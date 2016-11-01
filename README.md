# Neural IME:  Neural Input Method Engine
Japanese input method engine can enter next level with deep learning technology.

# Prerequisite
You need following software to use Neural IME.

* Python 3.5
* TensorFlow 0.10

The developer uses Mac OS X 10.11.4, Anaconda 4.1, PyCharm 5.0.4 but it should work elsewhere.


# Experimental results
The neural model outperformed N-gram model on reference corpus as shown below.

| Metrics             | N-gram | RNN       |
|:-------------------:|:------:|:---------:|
| Sentence Accuracy   | 41.5%  | 44.2%     |
| Prediction Accuracy | 22.9%  | **26.7%** |

# Training your own models
For training and testing your own models, you need annotated data
such as DVD version of BCCWJ (Balanced Corpus of Contemporary Written Japanese).

> http://pj.ninjal.ac.jp/corpus_center/bccwj/en/

You probably want a modern GPU to train faster, as the developer uses p2.xlarge instance in AWS.

## Preparing your data
Training data is text file in UTF-8 and each line corresponds to a sentence.
A sentence is segmented by space character into words, and a word is a pair of *target*
(i.e. Kanji, Hiragana or Katakana) and *source* (Hiragana), concatenated by slash character.

> 私/わたし の/の 名前/なまえ は/は 中野/なかの です/です 。/。

Test data should contain different sentences from training data, but ideally its domain is same to training data.
Source file should contain source sentences without space.

> きょうのてんきははれです。

Target file should contain target sentences without space.

> 今日の天気は晴れです。

## Pre-processing BCCWJ
The developer uses human-annotated part of BCCWJ as training and testing corpus.
You can use the scripts in this repository to pre-process the XML files after extracted from compressed file.
For example, the following commands parse and split data to train, test source and test target files.

    parse_bccwj.py 'BCCWJ/CORE/M-XML/*.xml' > parsed.txt

    split_train_test_valid.py parsed.txt train.txt test.txt valid.txt

    split_source_target.py test.txt test.target.txt test.source.txt


## Training neural models
Now you can train your own model with default parameters.

    train.py train.txt model

See help for optional parameters such as number of hidden units and dropout probability.

    train.py --help

## Decoding sentences
Once trained your model, you can decode sentences using it.

    decode.py model

Type source sentence on your console, it will show decoded sentence like this.

    きょうのてんきははれです。
    今日の天気は晴れです。
    きょじんにせんせい
    巨人に先制
 
Alternatively, you can give file names as input or output.

    decode.py model --input_file test.source.txt --output_file model/test.decode.txt

You can trade decoding time with accuracy by tuning pruning parameters such as beam size and viterbi size.
For example, the following option is faster than default beam size 5 but less accurate.

    decode.py model --beam_size 1

## Evaluating results
You can evaluate decoded results if you have target sentences as reference.

    evaluate.py model/test.decode.txt test.target.txt

This command gives something like this:

> precision: 93.59 recall: 93.58 f-score: 93.59 accuracy: 34.06

Precision, recall and F-score are character-based metrics based on longest common subsequence,
and accuracy is a sentence-level metric.

## Hyperparameter search
You can use grid search script to find best hyperparameters.

    grid_search.py train.txt valid.source.txt valid.target.txt model --hidden_size 50 100 200 400

## Training N-gram models
In order to train N-gram models as baseline for comparing with neural models, you need to install and use SRILM toolkit.

> http://www.speech.sri.com/projects/srilm/

Once installed, you can run the following command to train the model.

    ngram-count -text train.txt -lm ngram.txt -kndiscount -order 2

Now you can decode sentences using the N-gram model.

    decode_ngram.py ngram.txt

Or you can combine both neural model and N-gram model.

    decode_both.py neural_model ngram.txt


## Reference
> Yoh Okuno, Neural IME: Neural Input Method Engine, The 8th Input Method Workshop, 2016.
