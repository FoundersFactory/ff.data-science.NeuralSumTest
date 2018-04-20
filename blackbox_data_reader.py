# From data_reader.py

import os
import codecs
import numpy

from data_reader import Vocab


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
    char_vocab = Vocab()

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
                    a_char_list.append(char_vocab.feed(c))
                a_line_list.append(a_char_list)

            dom = a_line_list

        doms.append(dom)
        line_truths.append(line_truth)

    max_dom_length = max([len(dom) for dom in doms])
    max_line_length = max([len(line) for dom in doms for line in dom])

    # Now we have the sizes, create tensors
    line_tensor = numpy.zeros([len(doms), max_dom_length, max_line_length], dtype=numpy.int32)
    label_tensor = numpy.zeros([len(doms), max_dom_length], dtype=numpy.int32)

    for i, dom in enumerate(doms):
        for j, line in enumerate(dom):
            line_tensor[i][j][0:len(line)] = line

    print()
    print("Number of examples loaded:", len(doms))
    print("Maximum DOM length (lines):", max_dom_length)
    print("Maximum sentence length (chars):", max_line_length)

    return line_tensor, label_tensor, max_dom_length, max_line_length, char_vocab


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

        x_batches = numpy.transpose(x_batches, axes=(1, 0, 2, 3))
        y_batches = numpy.transpose(y_batches, axes=(1, 0, 2))
        self._x_batches = list(x_batches)
        self._y_batches = list(y_batches)

        assert len(self._x_batches) == len(self._y_batches)

        self.length = len(self._y_batches)
        self.batch_size = batch_size
        self.max_line_length = line_length

    def iter(self):
        for x, y in zip(self._x_batches, self._y_batches):
            yield x, y


if __name__ == '__main__':

    data_dir = '/home/ubuntu/data_store/training_data/10'
    master_file = 'main/main_10.csv'

    line_tensor, label_tensor, max_dom_length, max_line_length, char_vocab = load_data(data_dir, master_file, 10, 10)
   


