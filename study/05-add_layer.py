import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_function=None):
  Weights = tf.Variable(tf.random_normal([in_size, out_size]))  # 定义一个in_size 行  out_size列 的矩阵
  biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)           # 类似列表，不是矩阵给，推荐不为零， 所以+0.1
  Wx_plus_b = tf.matmul(inputs, Weights) + biases               # 预测的值存放  未被激活
  if activation_function is None:                               # 表示线性方法
    outputs = Wx_plus_b
  else:
    outputs = activation_function(Wx_plus_b)
  return outputs

# 定义数据形式   np.newaxis 在原来的数组上增加一个维度
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]                 # 有 300个例子 [:, np.newaxis] 列变成行既成二维 np.newaxis 同 None
noise = np.random.normal(0, 0.05, x_data.shape)                 # 噪点  不按照函数线走。分布误差出来，贴近现实
y_data = np.square(x_data) - 0.5 + noise                        #

xs = tf.placeholder(tf.float32,[None, 1])     # None 表示不确定多少行， 1 表示一列
ys = tf.placeholder(tf.float32,[None, 1])

# 定义一个 隐藏层 输入1 10个神经元，输出1
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# 定义 输出层
prediction = add_layer(l1, 10, 1, activation_function=None) 


#误差的平均值
# loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))

#练习减少误差 以0.1的效率
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)     #每一个联系的步骤都通过优化器，以每个误差0.1的效率来更正误差

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to see the step improvement
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
