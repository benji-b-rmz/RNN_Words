# Benjamin Ramirez April 11, 2017
# WebApp for generating a mini-story from Recurrent Neural Networks

import os
from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from model.lstm import StoryNet


# sequence length, number of words input to the net
# number of units in RNN cell
n_hidden = 256

full_path = os.path.dirname(os.path.realpath(__file__))
training_file = open(full_path + '/model/texts/hitch_hiker_quotes.txt', 'r')
sess = tf.Session()
training_text = training_file.read()
# alice net trained with 15 word input, 512 unit lstm cells, 3 lstm cells
test_net = StoryNet(training_text, sess, 'hh', 10, 15)
vocab_size, dictionary, reverse_dictionary = len(test_net.dictionary), test_net.dictionary, test_net.reverse_dictionary
n_input = 10

# tf Graph input
x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, vocab_size])
sess = tf.Session()
# RNN output node weights and biases

weights = test_net.weights
biases = test_net.biases


def RNN(x, weights, biases):

    x = tf.reshape(x, [-1, n_input])
    x = tf.split(x,n_input,1)
    rnn_cell = rnn.MultiRNNCell([
        rnn.BasicLSTMCell(n_hidden),
        rnn.BasicLSTMCell(n_hidden)
    ])
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    # we only want the last output
    return tf.matmul(outputs[-1], weights) + biases

pred = RNN(x, weights, biases)
# load the previously trained values
saver = tf.train.Saver()
saver.restore(sess, './model/checkpoints/hh.ckpt')


def create_story(input_sentence):

    sentence = input_sentence
    words = sentence.split(' ')

    if len(words) != n_input:
        print("Wrong number of words!, try %s" %n_input)
        return("Wrong number of words!, try: %s words please" % n_input)

    try:
        symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
        print(symbols_in_keys)
        for i in range(100):
            keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
            onehot_pred = sess.run(pred, feed_dict={x: keys})
            onehot_pred_index = int(np.argmax(onehot_pred))
            sentence = "%s %s" % (sentence, reverse_dictionary[onehot_pred_index])
            symbols_in_keys = symbols_in_keys[1:]
            symbols_in_keys.append(onehot_pred_index)
        print(sentence)
        return(sentence)

    except:
        return("Word not in dictionary!\nChoose words from the text")

# web application
app = Flask(__name__)


@app.route('/api/storygen', methods=['POST'])
def story():

    input = request.data.decode(encoding='UTF-8')
    return create_story(input)


@app.route('/')
def home():
    # send the dictionary of words to the template
    return render_template('index.html', hh_word_bank = reverse_dictionary.values())

if __name__ == "__main__":
    app.run(debug=True)

