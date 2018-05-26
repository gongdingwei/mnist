#!/usr/bin/env python
# encoding: utf-8
"""
tensorflow :generate my own dataset
@author: mengping
"""


from PIL import Image, ImageFilter
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from log_config import *
def imageprepare():
    input_images = np.array([0] * 336)
    file_name='./data/11/11_112.jpg'#导入自己的图片地址
    #in terminal 'mogrify -format png *.jpg' convert jpg to png
    img = Image.open(file_name).convert('L')
    width = img.size[0]
    height = img.size[1]
    for h in range(0, height):
        for w in range(0, width):
            # 通过这样的处理，使数字的线条变细，有利于提高识别准确率
            if img.getpixel((w, h)) > 60:
                input_images[w + h * width] = 0
            else:
                input_images[w + h * width] = 1
    return input_images

result = imageprepare()
x = tf.placeholder(tf.float32, [1,336])
# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
x_image = tf.reshape(x, [-1, 24, 14, 1])
# 定义第一个卷积层的variables和ops
W_conv1 = tf.Variable(tf.truncated_normal([3, 3, 1, 16], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[16]))

L1_conv = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
L1_relu = tf.nn.relu(L1_conv + b_conv1)
L1_pool = tf.nn.max_pool(L1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 定义第二个卷积层的variables和ops
W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 16, 32], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[32]))

L2_conv = tf.nn.conv2d(L1_pool, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
L2_relu = tf.nn.relu(L2_conv + b_conv2)
L2_pool = tf.nn.max_pool(L2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 全连接层
W_fc1 = tf.Variable(tf.truncated_normal([6 * 4 * 32, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

h_pool2_flat = tf.reshape(L2_pool, [ -1,6 * 4 * 32])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout层
W_fc2 = tf.Variable(tf.truncated_normal([1024, 12], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[12]))

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

init_op = tf.initialize_all_variables()



"""
Load the model2.ckpt file
file is stored in the same directory as this python script is started
Use the model to predict the integer. Integer is returend as list.

Based on the documentatoin at
https://www.tensorflow.org/versions/master/how_tos/variables/index.html
"""
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)
    saver.restore(sess, "./checkpoint_dir/MyModel/model.ckpt")#这里使用了之前保存的模型参数
    #print ("Model restored.")
    prediction=tf.argmax(y_conv,1)
    predint=prediction.eval(feed_dict={x: [result],keep_prob: 1.0}, session=sess)

    # print(h_conv2)
    print('recognize result:')
    # logger.debug('recognize result :  %s' % predint[0])
    # print('recognize result:')
    print(predint[0])