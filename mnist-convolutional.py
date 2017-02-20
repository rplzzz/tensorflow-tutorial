#!/usr/bin/env python

import tensorflow as tf

## Download and read in the benchmark data set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

### Define input nodes
with tf.name_scope("input"):
    ## Create a placeholder for the input images.  Each row in the ph is an image;
    ## each column is a pixel 
    x = tf.placeholder(tf.float32, [None, 784], name="data")
    ## Next we reshape the input into 28x28x1 (it was flattened before).  The new
    ## shape will be [N_image, 28, 28, 1].  We don't know how many images will be
    ## passed through the placeholder, so we use the special value -1 in the first
    ## dimension, which will cause that dimension to be computed at run time so as
    ## to hold all of the data passed in.
    ximg = tf.reshape(x, [-1, 28, 28, 1], name="images")
    ## The actual labels ("yhat").  Each is a vector of length 10.
    yhat = tf.placeholder(tf.float32, [None, 10], name="labels")


## Define some convenience functions for setting up nodes.  These give us a
## little bit of jitter in our initial values for symmetry breaking. Bias
## variables are initialized to a small positive value because apparently you
## can wind up with nodes that never activate if their values are stuck at
## zero.
def wgtvar(shape, name=None):
    iv = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(iv, name=name)

def bvar(shape, name=None):
    iv = tf.constant(0.1, shape=shape)
    return tf.Variable(iv, name)

## We will use a series of convolutional layers and max pooling.  Each
## convolutional layer looks at a compact block of pixels (e.g. a 5x5 patch) and
## computes one or more features from that block.  Max pooling downsamples the
## neighboring convolutional features (e.g., in a 2x2 block) by taking the
## largest activation value.  The following functions allow us generate these
## layers easily.
def conv2d(x, W, name=None):
    ## x: the input image data.  This tells us how big the output should be.  The
    ##    2d grid of convolved features will be the same size as the input
    ##    images.  The images will be zero-padded as necessary to make this
    ##    happen.
    ## W: the weight tensor.  The shape of this will tell us how big the viewing
    ##    patch will be and how many features will be computed for each patch
    ##    (this is also known as the "depth" of the layer).  Note that x and W
    ##    both have a dimension for the number of input channels.  These have to
    ##    match, or this step will fail.
    ## name: name tag for the operation.
    ## return: a graph node that computes the 2D convolution over the input
    ##         images using the specified weights.  For N images of size nx x ny
    ##         x nchan the output will be a tensor of shape [N, nx, ny, depth].
    return tf.nn.conv2d(x, W, [1,1,1,1], padding="SAME", name=name)

def maxpool2x2(x, name=None):
    ## x: features from the convolution layer.
    ## return: tensor 2x2 max-pooled results.  This should (I think) have size
    ##         [N,nx/2,ny/2,depth].  Odd nx or ny will round up.
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME",
                          name=name)

### Define conv-layer-1 nodes
with tf.name_scope("conv-layer-1"):
    ## For our first convolutional layer, we will do 5x5 patches with 32 features
    ## on each patch.  There is only one input channel per pixel (there would be
    ## more if we had, for example, multispectral data), so the shape of this layer
    ## is 5x5x1x32.  The weight tensor will have that dimension.  We'll use one of
    ## the convenience functions above to generate the weight tensor 
    Wconv1 = wgtvar([5,5,1,32], name="wgt")

    ## The bias will be applied to each feature (i.e., after the convolution).
    ## Also, the bias will be position independent, so we only have 32 bias values 
    bconv1 = bvar([32], name="bias")

    ## Now we convolve the input images with the weights, add the bias, and apply
    ## the activation function.  In this case we will use the ReLU for the latter.
    ## The results of this will be fed into the max_pool piece, and that's the end
    ## of this layer.
    hconv1 = tf.nn.relu(conv2d(ximg, Wconv1, name="conv") + bconv1, name="hconv")
    hpool1 = maxpool2x2(hconv1, name="hpool")

    ## XXX: As a side note, I'm still not sure why we are allowed to add a vector of
    ## biases to the tensor of convolution outputs.  My best guess is that for
    ## tensor addition there is some sort of default promotion from lower rank to
    ## higher rank, similar to how you can add a scalar to a vector in most vector
    ## libraries, but this isn't really explained in the tutorial.


with tf.name_scope("conv-layer-2"):    
    ## Now we'll stack a second convolutional layer on top of this.  We'll do
    ## another 5x5 convolution kernel, but this time we'll have 32 input channels
    ## (the 32 features computed by the previous layer).  We'll make 64 output
    ## channels from this layer.
    Wconv2 = wgtvar([5,5,32,64], name="wgt")
    bconv2 = bvar([64], name="bias")

    hconv2 = tf.nn.relu(conv2d(hpool1, Wconv2) + bconv2, name="hconv")
    hpool2 = maxpool2x2(hconv2, name="hpool")

with tf.name_scope("fc1"):
    ## That's enough convolution for this example.  Let's follow up with a
    ## densely-connected layer.  The second pooling layer produced an image size of
    ## 7x7, with 64 features.  We'll flatten that into a vector and feed it to a
    ## fully-connected layer with 1024 neurons.  We'll need weights and biases for
    ## each of those inputs.
    imglen = 7*7*64
    Wfc1 = wgtvar([imglen, 1024], name="wgt")
    bfc1 = bvar([1024], name="bias")

    hpool2f = tf.reshape(hpool2, [-1, imglen], name="hpool-flattened")
    ## as usual, we have to multiply the input from the left because we store
    ## samples in rows.
    hfc1 = tf.nn.relu(tf.matmul(hpool2f, Wfc1) + bfc1, name="hfc")

    ## A setup like we have above might be prone to overfitting, so we will apply
    ## dropout.  What this means is that we'll specify a probability p, and each
    ## neuron in the fc layer will be kept with probability p and dropped with
    ## probability 1-p, each time the network is evaluated.  You do this during
    ## training, but not in actual use, so we pass the dropout probability in as a
    ## placeholder.  This also allows us to experiment with different dropout
    ## probabilities. 
    pkeep = tf.placeholder(tf.float32, name="pkeep")
    hfc1drop = tf.nn.dropout(hfc1, pkeep, name="hfc1-dropout")

with tf.name_scope("output"):
    ## Finally, we need a readout layer.  This is going to take our 1024 nodes and
    ## reduce them down to a vector of 10 values representing the probabilities of
    ## the ten digits.  These are the linear predictors (now not so linear).  We'll
    ## pass them through a softmax to get probabilities.
    
    Wfc2 = wgtvar([1024,10], name="wgt")
    bfc2 = bvar([10], name="bias")

    y = tf.matmul(hfc1drop, Wfc2) + bfc2

## Now we're ready to train and evaluate.  We'll use the ADAM optimization
## algorithm because it's allegedly more better.  We'll need many more
## iterations to make this work, and it will take a while, so we'll log our
## progress at every 100th iteration.

hy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=yhat,
                                                             logits=y) )
train_step = tf.train.AdamOptimizer(1e-4).minimize(hy)
correct = tf.equal(tf.argmax(yhat, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

## define a function to package data for input into the placeholders
def feed_data(xin, yhatin, pkeepin):
    return {x: xin, yhat: yhatin, pkeep: pkeepin}

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(10000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        trn_accuracy = sess.run(accuracy, feed_dict=feed_data(batch[0], batch[1], 1.0))
        print("step %d:\ttraining accuracy: %g" % (i, trn_accuracy))
    sess.run(train_step, feed_dict=feed_data(batch[0], batch[1], 0.5))

test_accuracy = sess.run(accuracy, feed_dict=feed_data(mnist.test.images, mnist.test.labels, 1.0))
print("\ntest accuracy: %g" % test_accuracy)

    
