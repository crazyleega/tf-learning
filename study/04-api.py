import tensorflow as tf
import numpy as np


# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3   # 期望的参数

# create tensorflow structure start
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))   #随机初始化 -1到1 的一个数
biases = tf.Variable(tf.zeros([1]))   # 初始化0

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5) # 0.5为学习效率
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()   # 初始化变量

# crette tensorflow structure end

sess = tf.Session()  # 对话控制
sess.run(init)    # 激活神经网络 

for step in range(201):
  sess.run(train)
  if step % 20 == 0:
    print(step,sess.run(Weights),sess.run(biases)) 
    # print('value,sep=',end='\n',file=sys.stdout,flash=false)