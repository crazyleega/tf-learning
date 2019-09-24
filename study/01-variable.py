import tensorflow as tf

# print(tf.__version__)
state = tf.Variable(0, name='counter')
# print(state.name)
one = tf.constant(1)

new_val = tf.add(state, one)
update = tf.assign(state, new_val)

init = tf.global_variables_initializer()   # must have if define variable

with tf.Session() as sess:
  sess.run(init)
  for _ in range(3):
    sess.run(update)
    print(sess.run(state))   # 直接 打印 state不行