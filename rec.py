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
# def imageprepare():
#     input_images = np.array([[0] * 336])
#     file_name='./4_2125.jpg'#导入自己的图片地址
#     #in terminal 'mogrify -format png *.jpg' convert jpg to png
#     img = Image.open(file_name).convert('L')
#     width = img.size[0]
#     height = img.size[1]
#     for h in range(0, height):
#         for w in range(0, width):
#             # 通过这样的处理，使数字的线条变细，有利于提高识别准确率
#             if img.getpixel((w, h)) > 60:
#                 input_images[0][w + h * width] = 0
#             else:
#                 input_images[0][w + h * width] = 1
#
#
# # im.save("/home/mzm/MNIST_recognize/sample.png")
# # plt.imshow(im)
# # plt.show()
# # tv = list(img.getdata()) #get pixel values
# #
# # #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
# # tva = [ (x-60)*1.0/255.0 for x in tv]
# #print(tva)
#     return input_images
    # return tva

flag =0
cwd = os.getcwd()
classes = os.listdir(cwd + "/data")

for index, name in enumerate(classes):

    class_path = cwd + '/data/' + name + "/"
    if os.path.isdir(class_path):
        for img_name in os.listdir(class_path):
            input_images = np.array([[0] * 336])
            logger.debug('img_name :  %s' % img_name)
            img_path = class_path + img_name
            img = Image.open(img_path).convert('L')
            width = img.size[0]
            height = img.size[1]
            for h in range(0, height):
                for w in range(0, width):
                    # 通过这样的处理，使数字的线条变细，有利于提高识别准确率
                    if img.getpixel((w, h)) > 60:
                        input_images[0][w + h * width] = 0
                    else:
                        input_images[0][w + h * width] = 1

    # result = imageprepare()
            x = tf.placeholder(tf.float32, [1, 336])
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
                predint=prediction.eval(feed_dict={x: input_images,keep_prob: 1.0}, session=sess)

                # print(h_conv2)
                print('recognize result:')
                logger.debug('recognize result :  %s' % predint[0])
                # print('recognize result:')
                print(predint[0])

            tf.reset_default_graph()


