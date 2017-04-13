#testing storing and resoring tfVariables
import tensorflow as tf


print ("----------------Restoring NOT net-----------------")
# Truth Table as vectors/matrices, simply output opposite of the single input
not_inputs = [
    [-1.],
    [1.]
]

not_outputs = [
    [1.],
    [-1.]
]

x_ = tf.placeholder(tf.float32, shape=[2,1], name='X-inputs')
y_ = tf.placeholder(tf.float32, shape=[2,1], name='Y-outputs')

not_weights = tf.Variable(tf.random_normal([1,1]), name="weights")
not_bias = tf.Variable(tf.zeros([1]), name="biases")

#acivation function, tanh = scaled version of sigmoid,
hypothesis = tf.tanh(tf.add(tf.matmul( not_inputs, not_weights ) , not_bias))

error = tf.subtract(not_outputs, hypothesis)
cost = tf.reduce_mean(tf.square(error))
# the learning rate, alpha
alpha = 0.05

# using GradientDescentOptimizer as training algorithm, set learning rate
train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cost)

# add op to initialize variables
init = tf.global_variables_initializer()

# add ops to save and restore variables
saver = tf.train.Saver()

#max number of iterations if it fails to converge to target_error
current_error, target_error = 100, 0.00001
current_epoch, max_epochs = 0, 1500

#the weight should become negative, creating NOT operation
with tf.Session() as sess:

    #restore the trained weights
    saver.restore(sess, './checkpoints/model.ckpt')


    print('cost: ', sess.run(cost))
    print ('hypothesis: ', sess.run(hypothesis, feed_dict={x_: not_inputs, y_: not_outputs}))
    print('Weights: ', sess.run(not_weights))
    print('Bias: ', sess.run(not_bias))


    sess.close()
