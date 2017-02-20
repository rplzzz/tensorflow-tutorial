#!/Users/link593/pyenv/tf/bin/python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable bogus warning messages.
import tensorflow as tf


print('\nEx. 1')
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder = a+b                     # or tf.add(a,b)

sess = tf.Session()
print(sess.run(adder, {a:3, b:4.5}))
print(sess.run(adder, {a:[1,3], b:[2,4]}))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # restore logging once the bogus warnings are done

print('\nEx. 2')
add_and_3x = adder * 3.0
print(sess.run(add_and_3x, {a:3, b:4.5}))


print('\nEx. 3')
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([-0.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(linear_model, {x:[1,2,3,4]}))

y = tf.placeholder(tf.float32)
residsq = tf.square(linear_model - y)
loss = tf.reduce_sum(residsq)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

print('\nEx. 4')
updateW = tf.assign(W,[-1])
updateb = tf.assign(b, [1])
sess.run([updateW, updateb])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

print('\nEx. 5')
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init) # reset values to original values
for i in range(1000):           # Could we do this as something like: while loss > XYZ...
  sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
print(sess.run([W, b]))

file_writer = tf.summary.FileWriter('logs', sess.graph)
