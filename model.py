import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import mnist


def InterpolNet(images):
    # with tf.device('/cpu:0'):
    net = images

    with slim.arg_scope([slim.conv2d], padding='SAME'):

        net = slim.repeat(net, 3, slim.conv2d, 96, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        ne1 = net
        net = slim.repeat(net, 3, slim.conv2d, 96, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        ne2 = net

        net = slim.repeat(net, 3, slim.conv2d, 128, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        ne3 = net
        net = slim.repeat(net, 3, slim.conv2d, 128, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        ne4 = net
        net = slim.repeat(net, 3, slim.conv2d, 128, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')

        with slim.arg_scope([slim.conv2d_transpose], padding='SAME', stride=2):

            net = slim.conv2d_transpose(net, 128, [4, 4], scope='dcnv5')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='cnv5')
            net = tf.concat(3, [net, ne4])
            net = slim.conv2d_transpose(net, 128, [4, 4], scope='dcnv4')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='cnv4')
            net = tf.concat(3, [net, ne3])
            net = slim.conv2d_transpose(net, 128, [4, 4], scope='dcnv3')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='cnv3')

            net = tf.concat(3, [net, ne2])
            net = slim.conv2d_transpose(net, 96, [4, 4], scope='dcnv2')
            net = slim.repeat(net, 2, slim.conv2d, 96, [3, 3], scope='cnv2')
            net = tf.concat(3, [net, ne1])
            net = slim.conv2d_transpose(net, 96, [4, 4], scope='dcnv1')
            net = slim.repeat(net, 2, slim.conv2d, 96, [3, 3], scope='cnv1')

    net = slim.conv2d(net, 3, [3, 3], scope='cnv0')

    return net
