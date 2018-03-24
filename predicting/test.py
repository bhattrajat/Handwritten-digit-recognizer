import tensorflow as tf
import scipy.misc
import numpy as np
import sys

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def predict(x):
  x = np.array(x)
  #x = scipy.misc.imread(sys.argv[1])
  #x = (255-x)*1.0/255.0
  #x = np.reshape(x,(1,784))
  x_tensor = tf.convert_to_tensor(x,dtype=np.float32)
  x_tensor = tf.reshape(x_tensor, [-1, 28, 28, 1])
  sess=tf.Session()

  #First let's load meta graph and restore weights
  saver = tf.train.import_meta_graph('model.ckpt.meta')
  saver.restore(sess,tf.train.latest_checkpoint('./'))
  #saver.restore(sess,tf.train.latest_checkpoint("C:\\Users\\rajat\\Desktop\\paint\\data\\model.ckpt.data-00000-of-00001"))


  # Access saved Variables directly
  #print(sess.run('W_conv1:0'))
  # This will print 2, which is the value of bias that we saved


  # Now, let's access and create placeholders variables and
  # create feed-dict to feed new data

  graph = tf.get_default_graph()
  W_conv1 = graph.get_tensor_by_name("W_conv1:0")
  b_conv1 = graph.get_tensor_by_name("b_conv1:0")

  h_conv1 = tf.nn.relu(conv2d(x_tensor, W_conv1) + b_conv1)

  h_pool1 = max_pool_2x2(h_conv1)

  W_conv2 = graph.get_tensor_by_name("W_conv2:0")
  b_conv2 = graph.get_tensor_by_name("b_conv2:0")
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

  W_fc1 = graph.get_tensor_by_name("W_fc1:0")
  b_fc1 = graph.get_tensor_by_name("b_fc1:0")
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  W_fc2 = graph.get_tensor_by_name("W_fc2:0")
  b_fc2 = graph.get_tensor_by_name("b_fc2:0")
  y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
  y = tf.nn.softmax(y_conv)
  ans = tf.argmax(y,axis=1)
  print(sess.run(ans))
  '''
  feed_dict ={w1:13.0,w2:17.0}

  #Now, access the op that you want to run. 
  op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

  print sess.run(op_to_restore,feed_dict)
  #This will print 60 which is calculated 
  '''
