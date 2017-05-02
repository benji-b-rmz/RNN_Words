# Benjamin Ramirez April 29, 2017
# inspired by LSTM Example by Rowel Atienza:
# https://medium.com/towards-data-science/lstm-by-example-using-tensorflow-feb0c1968537
# extended to longer sequence inputs, trained on different dataset and different network architecture

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import pickle
import os
import json


class StoryNet(object):

    def __init__(self, story_text, sess, var_scope, input_length, output_length):

        self.var_scope = var_scope
        self.sess = sess
        # process and store the story data
        try:
            full_path = os.path.dirname(os.path.realpath(__file__))
            with open(full_path + "/texts/"+var_scope+"_dict.txt", "rb") as dict_file:
                self.dictionary = pickle.load(dict_file)
            with open(full_path + "/texts/"+var_scope+"_r_dict.txt", "rb") as reverse_dict:
                self.reverse_dictionary = pickle.load(reverse_dict)
            self.story_data, _, __ = self.build_dataset(story_text, False)
            print(var_scope + "dictionary exists")
        except:
            print("creating fresh dicts")
            self.story_data, self.dictionary, self.reverse_dictionary = self.build_dataset(story_text, True)

        print(self.dictionary)
        print(self.reverse_dictionary)

        self.vocab_size = len(self.dictionary)
        # create placeholders for feeding to network operations
        self.x = tf.placeholder("float", [None, input_length, 1])
        self.y = tf.placeholder("float", [None, self.vocab_size])
        # network parameters, weights, biases
        self.num_hidden = 256
        self.input_length = input_length
        self.output_length = output_length
        self.var_scope = var_scope


        with tf.variable_scope(var_scope):
            self.weights = tf.Variable(tf.random_normal([self.num_hidden, self.vocab_size]), name=var_scope+"_W")
            self.biases = tf.Variable(tf.random_normal([self.vocab_size]), name=var_scope+"_b")

    def rnn(self, x, weights, biases):
        # reshape to [1, n_input]
        x = tf.reshape(x, [-1, self.input_length])

        # Generate a n_input-element sequence of inputs
        # (eg. [had] [a] [general] -> [20] [6] [33])
        x = tf.split(x, self.input_length, 1)

        # 5-layer LSTM, each layer has num_hidden units.
        rnn_cell = rnn.MultiRNNCell([
            rnn.BasicLSTMCell(self.num_hidden),
            rnn.BasicLSTMCell(self.num_hidden)
        ])

        # generate prediction
        outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

        # there are n_input outputs but
        # we only want the last output
        return tf.matmul(outputs[-1], weights) + biases

    def train_model(self, iterations, display_iters, learning_rate):

        pred = self.rnn(self.x, self.weights, self.biases)
        # Loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.y))
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

        # Model evaluation
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # init op
        init = tf.global_variables_initializer()

        # let the training begin
        self.sess.run(init)
        step = 0
        offset = random.randint(0, self.input_length + 1)
        end_offset = self.input_length + 1
        acc_total = 0
        loss_total = 0

        while step < iterations:
            # Generate a minibatch. Add some randomness on selection process.
            if offset > (len(self.story_data) - end_offset):
                offset = random.randint(0, self.input_length + 1)

            symbols_in_keys = [[self.dictionary[str(self.story_data[i])]] for i in range(offset, offset + self.input_length)]
            symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, self.input_length, 1])

            symbols_out_onehot = np.zeros([self.vocab_size], dtype=float)
            symbols_out_onehot[self.dictionary[str(self.story_data[offset + self.input_length])]] = 1.0
            symbols_out_onehot = np.reshape(symbols_out_onehot, [1, -1])

            _, acc, loss, onehot_pred = self.sess.run([optimizer, accuracy, cost, pred], \
                                                      feed_dict={self.x: symbols_in_keys, self.y: symbols_out_onehot})
            loss_total += loss
            acc_total += acc
            if (step + 1) % display_iters == 0:
                print("Iter= " + str(step + 1) + ", Average Loss= " + \
                      "{:.6f}".format(loss_total / display_iters) + ", Average Accuracy= " + \
                      "{:.2f}%".format(100 * acc_total / display_iters))
                acc_total = 0
                loss_total = 0
                symbols_in = [self.story_data[i] for i in range(offset, offset + self.input_length)]
                symbols_out = self.story_data[offset + self.input_length]
                symbols_out_pred = self.reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
                print("%s - [%s] vs [%s]" % (symbols_in, symbols_out, symbols_out_pred))
            step += 1
            offset += (self.input_length + 1)
        print("Optimization Finished!")

        # save the trained values of the weights and biases
        full_path = os.path.dirname(os.path.realpath(__file__))
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, full_path + '/checkpoints/' + self.var_scope + '.ckpt',
                               write_meta_graph=False, write_state=False)

        print("Run on command line.")
        while True:
            prompt = "%s words: " % self.input_length
            sentence = input(prompt)
            sentence = sentence.strip()
            words = sentence.split(' ')
            if len(words) != self.input_length:
                continue
            try:
                symbols_in_keys = [self.dictionary[str(words[i])] for i in range(len(words))]
                for i in range(self.output_length):
                    keys = np.reshape(np.array(symbols_in_keys), [-1, self.input_length, 1])
                    onehot_pred = self.sess.run(pred, feed_dict={self.x: keys})
                    onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                    sentence = "%s %s" % (sentence, self.reverse_dictionary[onehot_pred_index])
                    symbols_in_keys = symbols_in_keys[1:]
                    symbols_in_keys.append(onehot_pred_index)
                print(sentence)
            except:
                print("Word not in dictionary")

    def create_story(self, input_string):

        sentence = input_string
        words = sentence.split(' ')

        pred = self.rnn(self.x, self.weights, self.biases)

        full_path = os.path.dirname(os.path.realpath(__file__))
        saver = tf.train.Saver()
        saver.restore(self.sess, full_path + '/checkpoints/' + self.var_scope + '.ckpt')
        if len(words) != self.input_length:
            print("Wrong number of words!, try %s" % self.input_length)
            return ("Wrong number of words!, try: %s words please" % self.input_length)

        try:
            symbols_in_keys = [self.dictionary[str(words[i])] for i in range(len(words))]
            print(symbols_in_keys)
            for i in range(200):
                keys = np.reshape(np.array(symbols_in_keys), [-1, self.input_length, 1])
                onehot_pred = self.sess.run(pred, feed_dict={self.x: keys})
                onehot_pred_index = int(np.argmax(onehot_pred))
                sentence = "%s %s" % (sentence, self.reverse_dictionary[onehot_pred_index])
                symbols_in_keys = symbols_in_keys[1:]
                symbols_in_keys.append(onehot_pred_index)
            print(sentence)
            return (sentence)
        except:
            print("check the input")
            return "Error in prediction function, check tensorflow session"

    def build_dataset(self, story_text, store_data=False):

        # remove/seperate special characters from story data
        story_text = story_text.lower()
        story_text = story_text.replace(',', ' , ')
        story_text = story_text.replace('.', ' . ')
        story_text = story_text.replace(':', ' : ')
        story_text = story_text.replace(';', ' ; ')
        story_text = story_text.replace('"', ' " ')
        story_text = story_text.replace('”', ' ')
        story_text = story_text.replace('“', ' ')
        story_text = story_text.replace('?', ' ? ')
        story_text = story_text.replace('!', ' ! ')
        story_text = story_text.replace('-', ' - ')
        story_text = story_text.replace('(', ' ( ')
        story_text = story_text.replace(')', ' ) ')
        story_text = story_text.replace('[', ' [ ')
        story_text = story_text.replace(']', ' ] ')
        story_text = story_text.replace('―', ' ')

        # break the text apart and store it into useful data structures
        story_data = story_text.split()
        count = collections.Counter(story_data).most_common()
        # count.sort()
        # print (count)
        word_dict = dict()
        for word, _ in count:
            word_dict[word] = len(word_dict)
        reverse_dict = dict(zip(word_dict.values(), word_dict.keys()))
        if store_data == True:
            full_path = os.path.dirname(os.path.realpath(__file__))[:-1]
            print("creating dictionary files")
            with open(full_path + "/texts/"+self.var_scope+"_dict.txt", "wb") as dict_file:
                pickle.dump(word_dict, dict_file, protocol=0)
            with open(full_path + "/texts/" + self.var_scope + "_r_dict.txt", "wb") as r_dict_file:
                pickle.dump(reverse_dict, r_dict_file, protocol=0)

        return story_data, word_dict, reverse_dict


if __name__ == "__main__":
    full_path = os.path.dirname(os.path.realpath(__file__))
    training_file = open(full_path + '/texts/hitch_hiker_quotes.txt', 'r')
    sess = tf.Session()
    training_text = training_file.read()
    # alice net trained with 15 word input, 512 unit lstm cells, 3 lstm cells
    test_net = StoryNet(training_text, sess, 'hh', 10, 15)
    # # hitch-hiker quotes trained with 10 word input sequence, 256 unit lstm units per cell, 2 lstm cells
    test_net.create_story("oh dear deep fook is this answer is the .")