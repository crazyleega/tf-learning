import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
# mnist = tf.keras.datasets.mnist.load_data();


def add_layer(inputs, in_size, out_size, activation_function=None):
  Weights = tf.Variable(tf.random_normal([in_size, out_size]))  # 定义一个in_size 行  out_size列 的矩阵
  biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)           # 类似列表，不是矩阵给，推荐不为零， 所以+0.1
  Wx_plus_b = tf.matmul(inputs, Weights) + biases               # 预测的值存放  未被激活
  if activation_function is None:                               # 表示线性方法
    outputs = Wx_plus_b
  else:
    outputs = activation_function(Wx_plus_b)
  return outputs

# tf.argmax(y_pre, 1)    找出y-pre(矩阵) 每个维度最大值的索引  
# tf.equal(a, b)         判断矩阵对应值是否相等  返回相同维度的矩阵，但是值是true ，false
# tf.cast(a, tf.float32) 类型转换函数 
# tf.reduce_mean(a)      求平均值  如: [1. 0. 0. 1.]  输出0.5
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})                        # 生成预测值
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))      # 预测值得最大值 和 真实值 的 差别
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))        # 计算这组数据的正确性
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})               # 得到的结果是百分比
    return result



# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None,784])     # 28 *28
ys = tf.placeholder(tf.float32, [None,10])

# add output layer
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)     #每一个联系的步骤都通过优化器，以每个误差0.5的效率来更正误差

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys})
  if i %50 == 0:
     print(compute_accuracy(
            mnist.test.images, mnist.test.labels))