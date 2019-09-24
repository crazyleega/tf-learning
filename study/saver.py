import tensorflow as tf

## 1 save
# w1 = tf.placeholder(tf.float32, name="w1")
# w2 = tf.placeholder(tf.float32, name="w2")
# b1 = tf.Variable(4.0,name="bias")

# w3 = tf.add(w1,w2)
# w4 = tf.multiply(w3,b1,name="result")

# with tf.Session() as sess:
#   sess.run(tf.global_variables_initializer())
#   saver = tf.train.Saver()   ## saver = tf.train.Saver([w1,w2]) 可以只保存两个参数。默认保存全部参数
#   print(sess.run(w4,feed_dict = {w1:3, w2:4}))
#   saver.save(sess,'./modelData/model',global_step=1000)    ## global_step 文件名后面增加的数字 如：model-1000


## 2 restore

with tf.Session() as sess: 
  saver = tf.train.import_meta_graph('./modelData/model-1000.meta')
  saver.restore(sess, tf.train.latest_checkpoint('./modelData'))
  
  graph = tf.get_default_graph()
  w1 = graph.get_tensor_by_name("w1:0")    # w1 表示张量名称，0表示第一个
  w2 = graph.get_tensor_by_name("w2:0")
  feed_dict = {w1: 15.0, w2: 5.0 }

  result = graph.get_tensor_by_name("result:0")

  print(sess.run(result,feed_dict))