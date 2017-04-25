# classification
数值型特征样本数据集，利用神经网络进行二分类
# _*_ coding:utf-8 _*_
__author__ = 'lixianwei'

import numpy as np
import tensorflow as tf
import os
os.chdir(r"C:\python3.5\data")
print(os.getcwd())

def read_data(file_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)

    defaults = [[0.] for _ in range(1140)]
    data = tf.decode_csv(value, defaults)

    return tf.stack(data[2:]), data[1]

def create_pipeline(filename, batch_size, num_epochs=None):
    file_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
    example, label = read_data(file_queue)

    min_after_dequeue = 1000
    capacity = min_after_dequeue + batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )

    return example_batch, label_batch

x_train_batch, y_train_batch = create_pipeline('train.csv', 50, num_epochs=1000)
x_test, y_test = create_pipeline('test.csv', 60)
global_step = tf.Variable(0, trainable=False)
learning_rate = 0.5

# Input
x = tf.placeholder(tf.float32, shape=[None,1138])
y = tf.placeholder(tf.int32, [None])

# 3层神经网络
#layer1
w1 = tf.Variable(tf.random_normal([1138, 512], stddev=0.5))#512 0.933 64：0.88
b1 = tf.Variable(tf.random_normal([512]))
output1 = tf.matmul(x, w1) + b1

#layer2
w2 = tf.Variable(tf.random_normal([512, 1024], stddev=.5))#1024
b2 = tf.Variable(tf.random_normal([1024]))
output2 = tf.nn.softmax(tf.matmul(output1, w2) + b2)

#layer3
w3 = tf.Variable(tf.random_normal([1024, 2], stddev=.5))
b3 = tf.Variable(tf.random_normal([2]))
output = tf.nn.softmax(tf.matmul(output2, w3) + b3)

# Training

cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=y))
tf.summary.scalar('Cross_Entropy', cross_entropy)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
correct_prediction = tf.equal(tf.argmax(output, 1), tf.cast(y, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('Accuracy', accuracy)

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
merged_summary = tf.summary.merge_all()

sess = tf.Session()
train_writer = tf.summary.FileWriter('logs/train', sess.graph)
test_writer = tf.summary.FileWriter('logs/test', sess.graph)
sess.run(init)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

try:
    print("Training: ")
    count = 0
    curr_x_test_batch, curr_y_test_batch = sess.run([x_test, y_test])
    while not coord.should_stop():
        # Run training steps or whatever
        curr_x_train_batch, curr_y_train_batch = sess.run([x_train_batch, y_train_batch])
        sess.run(train_step, feed_dict={
            x: curr_x_train_batch,
            y: curr_y_train_batch
        })

        count += 1
        ce, summary = sess.run([cross_entropy, merged_summary], feed_dict={
            x: curr_x_train_batch,
            y: curr_y_train_batch
        })

        train_writer.add_summary(summary, count)

        ce, test_acc, test_summary = sess.run([cross_entropy, accuracy, merged_summary], feed_dict={
            x: curr_x_test_batch,
            y: curr_y_test_batch
        })
        test_writer.add_summary(summary, count)
        print('Batch', count, 'J = ', ce, '测试准确率=', test_acc)
except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    # When done, ask the threads to stop.
    coord.request_stop()

# Wait for threads to finish.
coord.join(threads)
sess.close()

