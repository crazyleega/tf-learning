import tensorflow as tf

#  绑定feed_dict 传入参数至 run
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
  # 用了placeholder 就和 feed_dict 绑定传值
  print(sess.run(output,feed_dict={input1: [7.], input2: [2.]}))