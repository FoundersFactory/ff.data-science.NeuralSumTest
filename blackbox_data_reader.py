# From data_reader.py

from __future__ import print_function
from __future__ import division

import os
import codecs
import collections
import numpy as np


class Vocab:

    def __init__(self, token2index=None, index2token=None):
        self._token2index = token2index or {}
        self._index2token = index2token or []

    def feed(self, token):
        if token not in self._token2index:
            # allocate new index for this token
            index = len(self._token2index)
            self._token2index[token] = index
            self._index2token.append(token)

        return self._token2index[token]

    @property
    def size(self):
        return len(self._token2index)

    @property
    def token2index(self):
        return self._token2index

    def token(self, index):
        return self._index2token[index]

    def __getitem__(self, token):
        index = self.get(token)
        if index is None:
            raise KeyError(token)
        return index

    def get(self, token, default=None):
        return self._token2index.get(token, default)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self._token2index, self._index2token), f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            token2index, index2token = pickle.load(f)

        return cls(token2index, index2token)


def load_data(data_dir, a_master_file, max_doc_length, max_sen_length):
    """
    Load training data for CNN through following steps:
        i) Go through data directory, find master file containing list of filepaths & ground truths
        ii) Loop through filepaths, load each file, store file-level ground truths
        iii) Extract line-level ground truths from each file, clean html, store separately
        iv) Encode each html file on a character level
        v) Return encoded sentence lists, sentence ground truths, max sentence length

    :param data_dir: Str, path to root data directory
    :param a_master_file: Str, name of master file containing filepaths and ground truths
    :return: dict (sentences), dict (sentence-level truths), int (max sentence length)
    """
    doms = []
    line_truths = []
    word_vocab = Vocab()

    # Read master CSV listing paths and ground truths
    print('Reading', a_master_file)
    pname = os.path.join(data_dir, a_master_file)

    with codecs.open(pname, 'r', 'utf-8') as f:
        # Note - header line is skipped
        fpaths, dom_truths = zip(*(line.rstrip('\n').split(',') for line in f.readlines()[1:]))

    # Iterate through file paths, load files, lose empty strings
    for fpath in fpaths:
        with codecs.open(fpath, 'r', 'utf-8') as f:
            # Remove leading/trailing whitespace (isolating html)
            dom = [line.strip() for line in f.read().split('\n')]
            dom = list(filter(None, dom))
            dom = dom[:max_doc_length]

            # Separate out the ground truths
            line_truth = [line.split('\t\t\t')[1] for line in dom]
            dom = [line.split('\t\t\t')[0] for line in dom]

            # Character-encode every sentence (unicode ordering)
            a_line_list = list()
            for line in dom:
                a_char_list = list()
                for idx, c in enumerate(line):
                    if idx > max_sen_length -1:
                        break
                    a_char_list.append(word_vocab.feed(c))
                a_line_list.append(a_char_list)

            dom = a_line_list

        doms.append(dom)
        line_truths.append(line_truth)

    max_dom_length = max([len(dom) for dom in doms])
    max_line_length = max([len(line) for dom in doms for line in dom])

    # Now we have the sizes, create tensors
    line_tensor = np.zeros([len(doms), max_dom_length, max_line_length], dtype=np.int32)
    label_tensor = np.zeros([len(doms), max_dom_length], dtype=np.int32)

    for i, dom in enumerate(doms):
        for j, line in enumerate(dom):
            line_tensor[i][j][0:len(line)] = line

    print()
    print("Number of examples loaded:", len(doms))
    print("Maximum DOM length (lines):", max_dom_length)
    print("Maximum sentence length (chars):", max_line_length)

    return line_tensor, label_tensor, max_dom_length, max_line_length, word_vocab

class DOMReader:

    def __init__(self, line_tensor, label_tensor, batch_size):
        length = line_tensor.shape[0]

        dom_length = line_tensor.shape[1]
        line_length = line_tensor.shape[2]

        # Round down length to whole number of slices
        clipped_length = int(length / batch_size) * batch_size
        line_tensor = line_tensor[:clipped_length]
        label_tensor = label_tensor[:clipped_length]

        # Put n-dimensional tensors into (n + 1)-dimensional batch lists
        x_batches = line_tensor.reshape([batch_size, -1, dom_length, line_length])
        y_batches = label_tensor.reshape([batch_size, -1, dom_length])

        x_batches = np.transpose(x_batches, axes=(1, 0, 2, 3))
        y_batches = np.transpose(y_batches, axes=(1, 0, 2))
        self._x_batches = list(x_batches)
        self._y_batches = list(y_batches)

        assert len(self._x_batches) == len(self._y_batches)

        self.length = len(self._y_batches)
        self.batch_size = batch_size
        self.max_line_length = line_length

    def iter(self):
        for x, y in zip(self._x_batches, self._y_batches):
            yield x, y


class DataReader:

    def __init__(self, word_tensor, label_tensor, batch_size):
        length = word_tensor.shape[0]

        doc_length = word_tensor.shape[1]
        sent_length = word_tensor.shape[2]

        # round down length to whole number of slices

        clipped_length = int(length / batch_size) * batch_size
        word_tensor = word_tensor[:clipped_length]
        label_tensor = label_tensor[:clipped_length]

        x_batches = word_tensor.reshape([batch_size, -1, doc_length, sent_length])
        y_batches = label_tensor.reshape([batch_size, -1, doc_length])

        x_batches = np.transpose(x_batches, axes=(1, 0, 2, 3))
        y_batches = np.transpose(y_batches, axes=(1, 0, 2))

        self._x_batches = list(x_batches)
        self._y_batches = list(y_batches)
        assert len(self._x_batches) == len(self._y_batches)
        self.length = len(self._y_batches)
        self.batch_size = batch_size
        self.max_sent_length = sent_length

    def iter(self):
        for x, y in zip(self._x_batches, self._y_batches):
            yield x, y


def load_data_abs(data_dir, max_doc_length=10, max_sent_length=50, max_output_length=100, use_abs=True):
    '''
        data loader for generation models
        use_abs: When it is set to True, we use the human summaries as target;
                 otherwise we use the sentences labeled with 1 as target.
    '''

    word_vocab = Vocab()
    word_vocab.feed(' ')
    word_vocab.feed('{')
    word_vocab.feed('}')

    abs_vocab = Vocab()
    abs_vocab.feed(' ')
    abs_vocab.feed('{')
    abs_vocab.feed('}')

    actual_max_doc_length = 0
    actual_max_ext_length = 0
    actual_max_abs_length = 0

    word_tokens = collections.defaultdict(list)
    ext_output = collections.defaultdict(list)
    abs_output = collections.defaultdict(list)

    for fname in ('train', 'valid', 'test'):
        print('reading', fname)
        pname = os.path.join(data_dir, fname)
        for dname in os.listdir(pname):

            with codecs.open(os.path.join(pname, dname), 'r', 'utf-8') as f:
                lines = f.read().split('\n\n')
                word_doc = []
                ext_doc = []

                for line in lines[1].split('\n'):
                    line = line.strip()
                    line = line.replace('}', '').replace('{', '').replace('|', '')
                    line = line.replace('<unk>', ' | ')

                    sent, label = line.split('\t\t\t')
                    sent = sent.split(' ')

                    if len(sent) > max_sent_length - 2:  # space for 'start' and 'end' words
                        sent = sent[:max_sent_length - 2]

                    word_array = [word_vocab.feed(c) for c in ['{'] + sent + ['}']]

                    word_doc.append(word_array)

                    if label == '1':
                        ext_doc.extend(word_array[1:-1])

                    if len(word_doc) == max_doc_length:
                        break

                actual_max_doc_length = max(actual_max_doc_length, len(word_doc))

                word_tokens[fname].append(word_doc)

                if len(ext_doc) > max_output_length - 2:
                    ext_doc = ext_doc[:max_output_length - 2]

                ext_doc = [word_vocab['{']] + ext_doc + [word_vocab['}']]
                ext_output[fname].append(ext_doc)

                actual_max_ext_length = max(actual_max_ext_length, len(ext_doc))

                abs_doc = lines[2].replace('\n', ' ')
                abs_doc = abs_doc.split(' ')
                if len(abs_doc) > max_output_length - 2:
                    abs_doc = abs_doc[:max_output_length - 2]

                abs_doc = [abs_vocab.feed(c) for c in ['{'] + abs_doc + ['}']]
                abs_output[fname].append(abs_doc)

                actual_max_abs_length = max(actual_max_abs_length, len(abs_doc))

    assert actual_max_doc_length <= max_doc_length

    print()
    print('actual longest document length is:', actual_max_doc_length)
    print('size of word vocabulary:', word_vocab.size)
    print('number of tokens in train:', len(word_tokens['train']))
    print('number of tokens in valid:', len(word_tokens['valid']))
    print('number of tokens in test:', len(word_tokens['test']))

    # now we know the sizes, create tensors
    word_tensors = {}
    target_tensors = {}
    target_vocab = word_vocab
    actual_max_target_length = actual_max_ext_length

    if use_abs:
        target_vocab = abs_vocab
        actual_max_target_length = actual_max_abs_length

    for fname in ('train', 'valid', 'test'):
        word_tensors[fname] = np.zeros([len(word_tokens[fname]), actual_max_doc_length, max_sent_length],
                                       dtype=np.int32)
        target_tensors[fname] = np.zeros([len(ext_output[fname]), max_output_length], dtype=np.int32)

        for i, word_doc in enumerate(word_tokens[fname]):
            for j, word_array in enumerate(word_doc):
                word_tensors[fname][i][j][0:len(word_array)] = word_array

        if use_abs:
            for i, abs_doc in enumerate(abs_output[fname]):
                target_tensors[fname][i][0:len(abs_doc)] = abs_doc
        else:
            for i, ext_doc in enumerate(ext_output[fname]):
                target_tensors[fname][i][0:len(ext_doc)] = ext_doc

    return word_vocab, word_tensors, actual_max_doc_length, target_vocab, target_tensors, actual_max_target_length


class DataReader_abs:

    def __init__(self, word_tensor, target_tensor, batch_size):
        length = word_tensor.shape[0]

        doc_length = word_tensor.shape[1]
        sent_length = word_tensor.shape[2]

        output_length = target_tensor.shape[1]
        # round down length to whole number of slices

        clipped_length = int(length / batch_size) * batch_size
        word_tensor = word_tensor[:clipped_length]
        target_tensor = target_tensor[:clipped_length]

        x_batches = word_tensor.reshape([batch_size, -1, doc_length, sent_length])
        y_batches = target_tensor.reshape([batch_size, -1, output_length])

        x_batches = np.transpose(x_batches, axes=(1, 0, 2, 3))
        y_batches = np.transpose(y_batches, axes=(1, 0, 2))

        self._x_batches = list(x_batches)
        self._y_batches = list(y_batches)
        assert len(self._x_batches) == len(self._y_batches)
        self.length = len(self._y_batches)
        self.batch_size = batch_size
        self.max_sent_length = sent_length

    def iter(self):
        for x, y in zip(self._x_batches, self._y_batches):
            yield x, y


if __name__ == '__main__':

    data_dir = '/home/ubuntu/data_store/training_data/10'
    master_file = 'main/main_10.csv'

    line_tensor, label_tensor, max_dom_length, max_line_length = load_data(data_dir, master_file)
    #vocab, word_tensors, max_length, label_tensors = load_data('data/demo', 5, 10)

    count = 0
    for x, y in DataReader(word_tensors['valid'], label_tensors['valid'], 6).iter():
        count += 1
        print(x.shape, y.shape)
        if count > 0:
            break

    vocab, word_tensors, max_length, target_vocab, target_tensors, max_length_target = load_data_abs('data/demo', 5, 50,
                                                                                                     150, use_abs=False)
    count = 0
    for x, y in DataReader_abs(word_tensors['valid'], target_tensors['valid'], 6).iter():
        count += 1
        print(x.shape, y.shape, max_length_target)
        if count > 0:
            break



