#!/usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable bogus warning messages.
import tensorflow as tf

## Download and read in the benchmark data set.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

## Data is split into:
##   mnist.train:  training data set (50,000 samples)
##   mnist.test:  testing data set (10,000 samples)
##   mnist.validation:  validation data set (5,000 samples)

## each data set has images (e.g. mnist.train.images) with the input data and
## labels (e.g. mnist.train.labels) with the correct classification for the
## corresponding images.  The 28x28 pixel images are flattened into vectors of
## length 784.

## Create a placeholder for the input images.  Each row in the ph is an image;
## each column is a pixel 
x = tf.placeholder(tf.float32, [None, 784])

## We'll use a multinomial logistic regression for our classifier.  The linear
## predictor for this has two matrices of coefficients.  We're going to multiply
## images on the left, so we need 784 rows.  There are 10 possible categories,
## so we need 10 columns.  
beta = tf.Variable(tf.zeros([784, 10]))
## x*beta is a row vector, so alpha needs to be a vector too.
alpha = tf.Variable(tf.zeros([10]))

## the prediction vector is the "softmax" (i.e., logistic transform) of the
## linear predictor
## y = tf.nn.softmax(tf.matmul(x,beta) + alpha)
yl = tf.matmul(x,beta) + alpha   #  don't apply the softmax; it will be applied below.
y = tf.nn.softmax(yl)

## The actual labels ("yhat").  Each is a vector of length 10.
yhat = tf.placeholder(tf.float32, [None, 10])

## compute the cross-entropy; this will be our loss function.  Note that the
## dimensions are indexed from 0, so the reduction over dimension 1 in the inner
## reduce reduces over the *columns*, which is what we want because each column
## is a single sample.  However, see explanation below
##hy = tf.reduce_mean(-tf.reduce_sum(yhat * tf.log(y), axis=1))

## technically what we wrote above is right, but it's problematic when we have
## y=0 because the log blows up.  The package includes a function to compute all
## of this with appropriate protection for edge values
hy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=yhat, logits=yl))

## evaluate model performance.  Find the max over each observation (i.e., row)
correct = tf.equal(tf.argmax(yl,1), tf.argmax(yhat,1))
## and average over samples
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


## This section evaluates the graph for one chunk of data and dumps it for visualization
# batch_x, batch_yhat = mnist.train.next_batch(100)
# sess = tf.Session()
# tf.global_variables_initializer().run(session=sess)
# print(sess.run(hy, feed_dict={x:batch_x, yhat:batch_yhat}))
# file_writer = tf.summary.FileWriter('logs', sess.graph)


## Use the gradient descent algorithm for our iteration step in training
step = tf.train.GradientDescentOptimizer(0.5).minimize(hy)

## Initialize the variables we created above
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

## Set up a loop to run our trainer for 1000 steps
for i in range(10000):
    batch_x, batch_yhat = mnist.train.next_batch(100)
    sess.run(step, {x:batch_x, yhat:batch_yhat})

## Compute the accuracy for the testing set
print("Accuracy: ")
print(sess.run(accuracy, {x: mnist.test.images, yhat: mnist.test.labels}))
