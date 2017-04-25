
# _*_ coding:utf-8 _*_
__author__ = 'lixianwei'
"""
利用卷积神经网络进行的分类，目前来看，准确率处于0.9以下，跟classification.py中一般的神经网络，效果不是很好，
分析：cnn主要应用在图像或者文本这种相邻数据有相关关系的问题上，本例的数据集目前不知道业务上样本中相邻特征之间有没有关系，有待进一步研究和分析
结合业务，分析数据集的意义，再考虑使用cnn或者其他的分类算法
"""

import os
import pandas as pd
import tensorflow as tf
import numpy as np
os.chdir(r"C:\python3.5\data")
def read_data(file_name):
    x_train = pd.read_csv(file_name)
    x_train= x_train.values
    data = x_train[:, 2:1091]
    labels = x_train[:, 1:2]  # [1,0]
    # 把分类转为one-hot
    labels_tmp = []
    for label in labels:
        tmp = []
        if label[0] == [1.0]:
            tmp = [1.0, 0.0]
        else:
            tmp = [0.0, 1.0]
        labels_tmp.append(tmp)
    labels = np.array(labels_tmp)
    return data,labels

n_output_layer = 2
# 定义待训练的神经网络
def convolutional_neural_network(data):
    weights = {'w_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
               'w_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
               'w_fc': tf.Variable(tf.random_normal([9 * 9 * 64, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_output_layer]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_output_layer]))}

    data = tf.reshape(data, [-1, 33, 33, 1])

    conv1 = tf.nn.relu(
        tf.add(tf.nn.conv2d(data, weights['w_conv1'], strides=[1, 1, 1, 1], padding='SAME'), biases['b_conv1']))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv2 = tf.nn.relu(
        tf.add(tf.nn.conv2d(conv1, weights['w_conv2'], strides=[1, 1, 1, 1], padding='SAME'), biases['b_conv2']))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    fc = tf.reshape(conv2, [-1, 9 * 9 * 64])
    fc = tf.nn.relu(tf.add(tf.matmul(fc, weights['w_fc']), biases['b_fc']))

    # dropout剔除一些"神经元"
    # fc = tf.nn.dropout(fc, 0.8)// 防止过拟合

    output = tf.add(tf.matmul(fc, weights['out']), biases['out'])
    return output


# 每次使用100条数据进行训练
batch_size = 100

X = tf.placeholder('float', [None, 33 * 33])
Y = tf.placeholder('float')


# 使用数据训练神经网络
def train_neural_network(X, Y):
    predict = convolutional_neural_network(X)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)  # learning rate 默认 0.001

    epochs = 1
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        epoch_loss = 0
        x_train, y_train = read_data('train.csv')
        for epoch in range(epochs):
            for i in range(100):
                x_train_ = x_train[batch_size*i:batch_size*(i+1)]
                y_train_ = y_train[batch_size*i:batch_size*(i+1)]
                _, c = session.run([optimizer, cost_func], feed_dict={X: x_train_, Y: y_train_})
                #epoch_loss += c
            print(epoch, ' : ', c)
        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        x_test,y_test = read_data('test.csv')
        print('准确率: ', accuracy.eval({X: x_test, Y: y_test}))

train_neural_network(X, Y)
