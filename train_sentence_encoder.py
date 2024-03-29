# From pretrain.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf

import neural_summarizer
from blackbox_data_reader import load_data, DOMReader
from config import FLAGS

def run_test(session, m, data, batch_size, num_steps):
    """Runs the model on the given data."""

    costs = 0.0
    iters = 0
    state = session.run(m.initial_state)

    for step, (x, y) in enumerate(reader.dataset_iterator(data, batch_size, num_steps)):
        cost, state = session.run([m.cost, m.final_state], {
            m.input_data: x,
            m.targets: y,
            m.initial_state: state
        })

        costs += cost
        iters += 1

    return costs / iters


def load_wordvec(embedding_path, word_vocab):
    """loads pretrained word vectors"""

    initW = np.random.uniform(-0.25, 0.25, (char_vocab_size, FLAGS.word_embed_size))
    with open(embedding_path, "r") as f:
        for line in f:
            line = line.rstrip().split(' ')
            word, vec = line[0], line[1:]
            if word_vocab.token2index.has_key(word):
                initW[word_vocab[word]] = np.asarray([float(x) for x in vec])
    return initW


def sparse2dense(x, char_vocab_size):
    """converts a sparse input to a dense representation, for computing the reconstruction loss"""

    x_dense = np.zeros([x.shape[0], x.shape[1], char_vocab_size], dtype=np.int32)
    for i in xrange(x.shape[0]):
        for j in xrange(x.shape[1]):
            data_idx = x[i][j]
            x_dense[i][j][data_idx] = 1
    return x_dense


def build_model(char_vocab_size, train):
    """Build a training or inference graph, based on the model choice

    :param char_vocab_size: Int, size of character pool used to encode sentences
    """

    my_model = None

    if train:
        pretrained_emb = None

        if FLAGS.model_choice == 'bilstm':
            my_model = neural_summarizer.cnn_sen_enc(
                    char_vocab_size=char_vocab_size,
                    word_embed_size=FLAGS.word_embed_size,
                    batch_size=FLAGS.batch_size,
                    num_highway_layers=FLAGS.highway_layers,
                    max_line_length=FLAGS.max_line_length,
                    kernels=eval(FLAGS.kernels),
                    kernel_features=eval(FLAGS.kernel_features),
                    max_doc_length=FLAGS.max_doc_length,
                    pretrained=pretrained_emb)

            my_model.update(neural_summarizer.bilstm_doc_enc(my_model.input_cnn,
                                           batch_size=FLAGS.batch_size,
                                           num_rnn_layers=FLAGS.rnn_layers,
                                           rnn_size=FLAGS.rnn_size,
                                           max_doc_length=FLAGS.max_doc_length,
                                           dropout=FLAGS.dropout))

            my_model.update(neural_summarizer.label_prediction(my_model.enc_outputs))
            my_model.update(neural_summarizer.self_prediction(my_model.enc_outputs, char_vocab_size))
            my_model.update(neural_summarizer.loss_pretrain(my_model.plogits, FLAGS.batch_size, FLAGS.max_doc_length, char_vocab_size))
            my_model.update(neural_summarizer.training_graph(my_model.loss * FLAGS.max_doc_length,
                    FLAGS.learning_rate, FLAGS.max_grad_norm))

        elif FLAGS.model_choice == 'lstm':
            my_model = neural_summarizer.cnn_sen_enc(
                    char_vocab_size=char_vocab_size,
                    word_embed_size=FLAGS.word_embed_size,
                    batch_size=FLAGS.batch_size,
                    num_highway_layers=FLAGS.highway_layers,
                    max_line_length=FLAGS.max_line_length,
                    kernels=eval(FLAGS.kernels),
                    kernel_features=eval(FLAGS.kernel_features),
                    max_doc_length=FLAGS.max_doc_length,
                    pretrained=pretrained_emb)

            my_model.update(neural_summarizer.lstm_doc_enc(my_model.input_cnn,
                                           batch_size=FLAGS.batch_size,
                                           num_rnn_layers=FLAGS.rnn_layers,
                                           rnn_size=FLAGS.rnn_size,
                                           max_doc_length=FLAGS.max_doc_length,
                                           dropout=FLAGS.dropout))

            my_model.update(neural_summarizer.lstm_doc_dec(my_model.input_cnn, my_model.final_enc_state,
                                           batch_size=FLAGS.batch_size,
                                           num_rnn_layers=FLAGS.rnn_layers,
                                           rnn_size=FLAGS.rnn_size,
                                           max_doc_length=FLAGS.max_doc_length,
                                           dropout=FLAGS.dropout))

            my_model.update(neural_summarizer.label_prediction_att(my_model.enc_outputs, my_model.dec_outputs))
            my_model.update(neural_summarizer.self_prediction(my_model.enc_outputs, char_vocab_size))
            my_model.update(neural_summarizer.loss_pretrain(my_model.plogits, FLAGS.batch_size, FLAGS.max_doc_length, char_vocab_size))

            my_model.update(neural_summarizer.training_graph(my_model.loss * FLAGS.max_doc_length,
                    FLAGS.learning_rate, FLAGS.max_grad_norm))

    else:
        if FLAGS.model_choice == 'bilstm':
            my_model = neural_summarizer.cnn_sen_enc(
                    char_vocab_size=char_vocab_size,
                    word_embed_size=FLAGS.word_embed_size,
                    batch_size=FLAGS.batch_size,
                    num_highway_layers=FLAGS.highway_layers,
                    max_line_length=FLAGS.max_line_length,
                    kernels=eval(FLAGS.kernels),
                    kernel_features=eval(FLAGS.kernel_features),
                    max_doc_length=FLAGS.max_doc_length)

            my_model.update(neural_summarizer.bilstm_doc_enc(my_model.input_cnn,
                                           batch_size=FLAGS.batch_size,
                                           num_rnn_layers=FLAGS.rnn_layers,
                                           rnn_size=FLAGS.rnn_size,
                                           max_doc_length=FLAGS.max_doc_length,
                                           dropout=FLAGS.dropout))

            my_model.update(neural_summarizer.self_prediction(my_model.enc_outputs, char_vocab_size))
            my_model.update(neural_summarizer.loss_pretrain(my_model.plogits, FLAGS.batch_size, FLAGS.max_doc_length, char_vocab_size))

        elif FLAGS.model_choice == 'lstm':
            my_model = neural_summarizer.cnn_sen_enc(
                    char_vocab_size=char_vocab_size,
                    word_embed_size=FLAGS.word_embed_size,
                    batch_size=FLAGS.batch_size,
                    num_highway_layers=FLAGS.highway_layers,
                    max_line_length=FLAGS.max_line_length,
                    kernels=eval(FLAGS.kernels),
                    kernel_features=eval(FLAGS.kernel_features),
                    max_doc_length=FLAGS.max_doc_length)

            my_model.update(neural_summarizer.lstm_doc_enc(my_model.input_cnn,
                                           batch_size=FLAGS.batch_size,
                                           num_rnn_layers=FLAGS.rnn_layers,
                                           rnn_size=FLAGS.rnn_size,
                                           max_doc_length=FLAGS.max_doc_length,
                                           dropout=FLAGS.dropout))

            my_model.update(neural_summarizer.lstm_doc_dec(my_model.input_cnn, my_model.final_enc_state,
                                           batch_size=FLAGS.batch_size,
                                           num_rnn_layers=FLAGS.rnn_layers,
                                           rnn_size=FLAGS.rnn_size,
                                           max_doc_length=FLAGS.max_doc_length,
                                           dropout=FLAGS.dropout))

            my_model.update(neural_summarizer.self_prediction(my_model.enc_outputs, char_vocab_size))
            my_model.update(neural_summarizer.loss_pretrain(my_model.plogits, FLAGS.batch_size, FLAGS.max_doc_length, char_vocab_size))
    return my_model


def main(_):
    """Train model from data"""

    if not os.path.exists(FLAGS.train_dir):
        os.mkdir(FLAGS.train_dir)
        print('Created training directory', FLAGS.train_dir)

    line_tensor, label_tensor, max_dom_length, max_line_length = load_data(
        FLAGS.data_dir,
        FLAGS.master_file
    )

    char_vocab_size = max([x for dom in line_tensor for line in dom for x in line])

    train_reader = DOMReader(line_tensor, label_tensor, FLAGS.batch_size)

    print('Initialized all dataset readers')

    with tf.Graph().as_default(), tf.Session() as session:

        # Tensorflow seed must be inside graph
        tf.set_random_seed(FLAGS.seed)
        np.random.seed(seed=FLAGS.seed)

        # build training graph
        initializer = tf.random_uniform_initializer(-FLAGS.param_init, FLAGS.param_init)

        with tf.variable_scope("Model", initializer=initializer):
            train_model = build_model(char_vocab_size, train=True)

        # create saver before creating more graph nodes, so that we do not save any vars defined below
        saver = tf.train.Saver(max_to_keep=50)

        # build graph for validation and testing (shares parameters with the training graph!)
        with tf.variable_scope("Model", reuse=True):
            valid_model = build_model(char_vocab_size, train=False)

        if FLAGS.load_model:
            saver.restore(session, FLAGS.load_model)
            print('Loaded model from', FLAGS.load_model, 'saved at global step', train_model.global_step.eval())
        else:
            tf.global_variables_initializer().run()
            session.run(train_model.clear_word_embedding_padding)
            print('Created and initialized fresh model. Size:', neural_summarizer.model_size())

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=session.graph)

        # take learning rate from CLI, not from saved graph
        session.run(
            tf.assign(train_model.learning_rate, FLAGS.learning_rate),
        )

        # training starts here
        best_valid_loss = None
        #rnn_state = session.run(train_model.initial_rnn_state)

        for epoch in range(FLAGS.max_epochs):

            epoch_start_time = time.time()
            avg_train_loss = 0.0
            count = 0
            for x, _ in train_reader.iter():
                y = sparse2dense(x, char_vocab_size)
                count += 1
                start_time = time.time()

                loss, _, gradient_norm, step, _ = session.run([
                    train_model.loss,
                    train_model.train_op,
                    train_model.global_norm,
                    train_model.global_step,
                    train_model.clear_word_embedding_padding
                ], {
                    train_model.input  : x,
                    train_model.targets: y,
                })

                avg_train_loss += 0.05 * (loss - avg_train_loss)

                time_elapsed = time.time() - start_time

                if count % FLAGS.print_every == 0:
                    print('%6d: %d [%5d/%5d], train_loss/perplexity = %6.8f/%6.7f secs/batch = %.4fs, grad.norm=%6.8f' % (step,
                                                            epoch, count,
                                                            train_reader.length,
                                                            loss, np.exp(loss),
                                                            time_elapsed,
                                                            gradient_norm))

            print('Epoch training time:', time.time()-epoch_start_time)

            # epoch done: time to evaluate
            avg_valid_loss = 0.0
            count = 0
            #rnn_state = session.run(valid_model.initial_rnn_state)
            for x, _ in valid_reader.iter():
                y = sparse2dense(x, char_vocab_size)
                count += 1
                start_time = time.time()

                loss = session.run(
                    valid_model.loss
                , {
                    valid_model.input  : x,
                    valid_model.targets: y,
                })

                if count % FLAGS.print_every == 0:
                    print("\t> validation loss = %6.8f, perplexity = %6.8f" % (loss, np.exp(loss)))
                avg_valid_loss += loss / valid_reader.length

            print("at the end of epoch:", epoch)
            print("train loss = %6.8f, perplexity = %6.8f" % (avg_train_loss, np.exp(avg_train_loss)))
            print("validation loss = %6.8f, perplexity = %6.8f" % (avg_valid_loss, np.exp(avg_valid_loss)))

            save_as = '%s/epoch%03d_%.4f.model' % (FLAGS.train_dir, epoch, avg_valid_loss)
            saver.save(session, save_as)
            print('Saved model', save_as)

            # write out summary events
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="train_loss", simple_value=avg_train_loss),
                tf.Summary.Value(tag="valid_loss", simple_value=avg_valid_loss)
            ])
            summary_writer.add_summary(summary, step)

            # decide if need to decay learning rate
            if best_valid_loss is not None and np.exp(avg_valid_loss) > np.exp(best_valid_loss) - FLAGS.decay_when:
                print('validation perplexity did not improve enough, decay learning rate')
                current_learning_rate = session.run(train_model.learning_rate)
                print('learning rate was:', current_learning_rate)
                current_learning_rate *= FLAGS.learning_rate_decay
                if current_learning_rate < 1.e-5:
                    print('learning rate too small - stopping now')
                    break

                session.run(train_model.learning_rate.assign(current_learning_rate))
                print('new learning rate is:', current_learning_rate)
            else:
                best_valid_loss = avg_valid_loss


if __name__ == "__main__":
    tf.app.run()
