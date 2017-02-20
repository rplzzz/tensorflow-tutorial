#!/Users/link593/pyenv/tf/bin/python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable bogus warning messages.
import tensorflow as tf

W = tf.Variable([0.3], tf.float32)
b = tf.Variable([-0.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

y = tf.placeholder(tf.float32)
residsq = tf.square(linear_model - y)
loss = tf.reduce_sum(residsq)


optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init) # reset values to original values
for i in range(1000):           # Could we do this as something like: while loss > XYZ...
    sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]}  )
print(sess.run([W, b]))

file_writer = tf.summary.FileWriter('logs', sess.graph)
