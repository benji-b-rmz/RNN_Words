# Benjamin Ramirez April 11, 2017
# WebApp for generating a mini-story from Recurrent Neural Networks


from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections


training_file = open('./model/texts/hitch_hiker_quotes.txt','r')

# prepping the file for input into training data
file_data = training_file.read()
file_data = file_data.replace(',', ' , ')
file_data = file_data.replace('.', ' . ')
file_data = file_data.replace(':', ' : ')
file_data = file_data.replace(';', ' ; ')
file_data = file_data.replace('"', ' " ')
file_data = file_data.replace('”', ' ” ')
file_data = file_data.replace('“', ' “ ')
file_data = file_data.replace('?', ' ? ')
file_data = file_data.replace('!', ' ! ')
file_data = file_data.replace('-', ' - ')
file_data = file_data.replace('(', ' ( ')
file_data = file_data.replace(')', ' ) ')
file_data = file_data.replace('[', ' [ ')
file_data = file_data.replace(']', ' ] ')
file_data = file_data.replace('―', ' ')

#store the words in the file in training data
training_data = file_data.split()
print("Loaded training data...")

def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

dictionary, reverse_dictionary = build_dataset(training_data)
vocab_size = len(dictionary)

# sequence length, number of words input to the net
n_input = 10

# number of units in RNN cell
n_hidden = 500

# tf Graph input
x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, vocab_size])

sess = tf.Session()
# RNN output node weights and biases
with tf.variable_scope("hh_guide"):
    weights = tf.Variable(tf.random_normal([n_hidden, vocab_size]), name="hh_weights")
    biases = tf.Variable(tf.random_normal([vocab_size]), name="hh_biases")

def RNN(x, weights, biases):

    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x,n_input,1)

    # 2-layer LSTM, each layer has n_hidden units.
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights) + biases

pred = RNN(x, weights, biases)

#load the previously trained values
saver = tf.train.Saver()
saver.restore(sess, './model/checkpoints/hh_model.ckpt' )

def create_story(input_sentence):

    sentence = input_sentence
    words = sentence.split(' ')

    if len(words) != n_input:
        print("Wrong number of words!, try %s" %n_input)
        return("Wrong number of words!, try: %s words please" % n_input)

    try:
        symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
        print(symbols_in_keys)
        for i in range(200):
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
    return render_template('index.html')

