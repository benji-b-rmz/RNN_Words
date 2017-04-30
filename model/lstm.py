# Benjamin Ramirez April 29, 2017
# inspired by LSTM Example by Rowel Atienza:
# https://medium.com/towards-data-science/lstm-by-example-using-tensorflow-feb0c1968537
# extended to longer sequence inputs, trained on different dataset and different network architecture

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections


class StoryNet(object):

    def __init__(self, story_text, var_scope, input_length, output_length):
        print("initializing Story Generating Network")
        # process and store the story data
        self.story_data, self.dictionary, self.reverse_dictionary = self.build_dataset(story_text)
        self.vocab_size = len(self.dictionary)
        # create placeholders for feeding to network operations
        self.x = tf.placeholder("float", [None, input_length, 1])
        self.y = tf.placeholder("float", [None, self.vocab_size])
        # network parameters, weights, biases
        self.num_hidden = 512
        self.input_length = input_length
        self.output_length = output_length
        self.var_scope = var_scope
        with tf.variable_scope(var_scope):
            self.weights = tf.Variable(tf.random_normal([self.num_hidden, self.vocab_size]), name=var_scope+"_W")
            self.biases = tf.Variable(tf.random_normal([self.vocab_size]), name=var_scope+"_b")

    def RNN(self, x, weights, biases):
        # reshape to [1, n_input]
        x = tf.reshape(x, [-1, self.input_length])

        # Generate a n_input-element sequence of inputs
        # (eg. [had] [a] [general] -> [20] [6] [33])
        x = tf.split(x, self.input_length, 1)

        # 5-layer LSTM, each layer has num_hidden units.
        rnn_cell = rnn.MultiRNNCell([
            rnn.BasicLSTMCell(self.num_hidden),
            rnn.BasicLSTMCell(self.num_hidden),
            rnn.BasicLSTMCell(self.num_hidden),
            rnn.BasicLSTMCell(self.num_hidden)
        ])

        # generate prediction
        outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

        # there are n_input outputs but
        # we only want the last output
        return tf.matmul(outputs[-1], weights) + biases

    def train_model(self, iterations, display_iters, learning_rate):

        pred = self.RNN(self.x, self.weights, self.biases)
        # Loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.y))
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

        # Model evaluation
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # init op
        init = tf.global_variables_initializer()

        # create Saver object to store Biases and Weights
        saver = tf.train.Saver()

        # let the training begin
        with tf.Session() as sess:
            sess.run(init)
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

                _, acc, loss, onehot_pred = sess.run([optimizer, accuracy, cost, pred], \
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
            save_path = saver.save(sess, '/checkpoints/' + self.var_scope + '.ckpt')

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
                        onehot_pred = sess.run(pred, feed_dict={self.x: keys})
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
        pred = self.RNN(self.x, self.weights, self.biases)

        with tf.Session() as sess:
            if len(words) != self.input_length:
                print("Wrong number of words!, try %s" % self.input_length)
                return ("Wrong number of words!, try: %s words please" % self.input_length)

            try:
                symbols_in_keys = [self.dictionary[str(words[i])] for i in range(len(words))]
                print(symbols_in_keys)
                for i in range(200):
                    keys = np.reshape(np.array(symbols_in_keys), [-1, self.input_length, 1])
                    onehot_pred = sess.run(pred, feed_dict={self.x: keys})
                    onehot_pred_index = int(np.argmax(onehot_pred))
                    sentence = "%s %s" % (sentence, self.reverse_dictionary[onehot_pred_index])
                    symbols_in_keys = symbols_in_keys[1:]
                    symbols_in_keys.append(onehot_pred_index)
                print(sentence)
                return (sentence)

            except:
                return ("Word not in dictionary!\nChoose words from the text")

    def build_dataset(self, story_text):

        # remove/seperate special characters from story data
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
        word_dict = dict()
        for word, _ in count:
            word_dict[word] = len(word_dict)
        reverse_dict = dict(zip(word_dict.values(), word_dict.keys()))

        return story_data, word_dict, reverse_dict


if __name__ == "__main__":
    training_file = open('./texts/hitch_hiker_quotes.txt', 'r')
    training_text = training_file.read()
    test_net = StoryNet(training_text, 'test', 5, 20)
    # sess = tf.Session()
    # saver = tf.train.Saver()
    # saver.restore(sess, '/checkpoints/test.ckpt')
    # test_net.create_story("Oh dear dear dear dear")
    test_net.train_model(1000, 100, 0.001)