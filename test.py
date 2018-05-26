#!/usr/bin/env python
# encoding: utf-8
"""
tensorflow :generate my own dataset
@author: mengping
"""

import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from PIL import Image
from log_config import *


# 第一次遍历图片目录是为了获取图片总数
input_count = 0
for i in range(0, 12):
    dir = './data/%s/' % i  # 这里可以改成你自己的图片目录，i为分类标签
    for rt, dirs, files in os.walk(dir):
        for filename in files:
            input_count += 1

            # 定义对应维数和各维长度的数组
input_images = np.array([[0] * 336 for i in range(input_count)])
input_labels = np.array([[0] * 12 for i in range(input_count)])

# 第二次遍历图片目录是为了生成图片数据和标签
index = 0
for i in range(0, 12):
    dir = './data/%s/' % i  # 这里可以改成你自己的图片目录，i为分类标签
    for rt, dirs, files in os.walk(dir):
        for filename in files:
            filename = dir + filename
            # print filename
            img = Image.open(filename)
            img = Image.open(filename).convert('L')
            width = img.size[0]
            height = img.size[1]
            for h in range(0, height):
                for w in range(0, width):
                    # 通过这样的处理，使数字的线条变细，有利于提高识别准确率
                    if img.getpixel((w, h)) > 60:
                        input_images[index][w + h * width] = 0
                    else:
                        input_images[index][w + h * width] = 1
            input_labels[index][i] = 1
            # print i
            index += 1
            # print len(input_images)

# 第一次遍历图片目录是为了获取图片总数
test_count = 0
for i in range(0, 12):
    dir = './data_test/%s/' % i  # 这里可以改成你自己的图片目录，i为分类标签
    for rt, dirs, files in os.walk(dir):
        for filename in files:
            test_count += 1

            # 定义对应维数和各维长度的数组
test_images = np.array([[0] * 336 for i in range(test_count)])
test_labels = np.array([[0] * 12 for i in range(test_count)])

# 第二次遍历图片目录是为了生成图片数据和标签
index = 0
for i in range(0, 12):
    dir = './data_test/%s/' % i  # 这里可以改成你自己的图片目录，i为分类标签
    for rt, dirs, files in os.walk(dir):
        for filename in files:
            filename = dir + filename
            # print filename
            img = Image.open(filename)
            img = Image.open(filename).convert('L')
            width = img.size[0]
            height = img.size[1]
            for h in range(0, height):
                for w in range(0, width):
                    # 通过这样的处理，使数字的线条变细，有利于提高识别准确率
                    if img.getpixel((w, h)) > 60:
                        test_images[index][w + h * width] = 0
                    else:
                        test_images[index][w + h * width] = 1
            test_labels[index][i] = 1
            # print i
            index += 1
                # print len(input_images)
# 定义输入节点，对应于图片像素值矩阵集合和图片标签(即所代表的数字)
x = tf.placeholder(tf.float32, shape=[None, 336])
y_ = tf.placeholder(tf.float32, shape=[None, 12])

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



batch_size = 60
iterations = 10000
batches_count = int(input_count / batch_size)
remainder = input_count % batch_size

# 定义优化器和训练op
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(1e-3,global_step,decay_steps=batches_count,decay_rate=0.98,staircase=True)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer((learning_rate)).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print ("一共读取了 %s 个输入图像， %s 个标签" % (input_count, input_count))

    # 设置每次训练op的输入个数和迭代次数，这里为了支持任意图片总数，定义了一个余数remainder，譬如，如果每次训练op的输入个数为60，图片总数为150张，则前面两次各输入60张，最后一次输入30张（余数30）

    print ("数据集分成 %s 批, 前面每批 %s 个数据，最后一批 %s 个数据" % (batches_count + 1, batch_size, remainder))
    saver = tf.train.Saver()
    # 执行训练迭代
    for it in range(iterations):
        # 这里的关键是要把输入数组转为np.array
        for n in range(batches_count):

            train_step.run(feed_dict={x: input_images[n * batch_size:(n + 1) * batch_size],y_: input_labels[n * batch_size:(n + 1) * batch_size], keep_prob: 0.5})

        if remainder > 0:
            start_index = batches_count * batch_size
            train_step.run(
                feed_dict={x: input_images[start_index:input_count - 1], y_: input_labels[start_index:input_count - 1],
                           keep_prob: 0.5})

            # 每完成五次迭代，判断准确度是否已达到100%，达到则退出迭代循环
        iterate_accuracy = 0
        if it % 5 == 0:
            iterate_accuracy = accuracy.eval(feed_dict={x: input_images, y_: input_labels, keep_prob: 1.0})
            print ('iteration %d: accuracy %s' % (it, iterate_accuracy))
            logger.debug('iteration %d: accuracy %s' % (it, iterate_accuracy))
        if it % 20 ==0:
            print "accuracy:", accuracy.eval(feed_dict={x: test_images, y_: test_labels,keep_prob: 1.0})

            logger.debug('test_iteration %d: accuracy %s' % (it, accuracy))
            saver.save(sess, './checkpoint_dir/MyModel/model.ckpt')
# if iterate_accuracy >= 1:
#     break;

# print ('完成训练!')