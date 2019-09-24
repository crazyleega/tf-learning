import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
  layer_name = 'layer%s' % n_layer
  with tf.name_scope(layer_name):
    with tf.name_scope('weights'):
      Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='w')  # 定义一个in_size 行  out_size列 的矩阵
      tf.summary.histogram(layer_name+'/weights',Weights)
    with tf.name_scope('biases'):
      biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')           # 类似列表，不是矩阵给，推荐不为零， 所以+0.1
      tf.summary.histogram(layer_name+'/biases',biases)
    with tf.name_scope('Wx_plus_b'):
      Wx_plus_b = tf.matmul(inputs, Weights) + biases               # 预测的值存放  未被激活
    if activation_function is None:                               # 表示线性方法
      outputs = Wx_plus_b
    else:
      outputs = activation_function(Wx_plus_b)
      tf.summary.histogram(layer_name+'/outputs',outputs)
    return outputs

# 定义数据形式
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]                 # 有 300个例子
noise = np.random.normal(0, 0.05, x_data.shape)                 # 噪点
y_data = np.square(x_data) - 0.5 + noise                        #


with tf.name_scope('inputs'):
  xs = tf.placeholder(tf.float32,[None, 1],name='x_input')
  ys = tf.placeholder(tf.float32,[None, 1],name='y_input')

# 定义一个 影藏层 输入1 10个神经元，输出1
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
# 定义 输出层
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None) 


#误差的平均值
with tf.name_scope('loss'):
  loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))
  tf.summary.scalar('loss',loss)

#练习减少误差 以0.1的效率
with tf.name_scope('train'):
  train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)     #每一个联系的步骤都通过优化器，以每个误差0.1的效率来更正误差


with tf.Session() as sess:
  merged = tf.summary.merge_all()    # 打包加的记录
  writer = tf.summary.FileWriter("logs/", sess.graph)
  init = tf.global_variables_initializer()
  sess.run(init)
  for i in range(1000):
    sess.run(train_step,feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
      result = sess.run(merged,feed_dict={xs: x_data, ys: y_data})
      writer.add_summary(result,i)                                    #每个i步添加一个结果



##  tensorboard --logdir /Users/Mac/Documents/Code/python/tensorflow/logs  --host localhost --port 8088