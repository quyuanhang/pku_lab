# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 10:12:12 2017

@author: QYH
"""
import tensorflow as tf
import pandas as pd

sess = tf.InteractiveSession()
data = pd.read_csv('input/ratMatrixTrain.csv',index_col=0)
data_T = data.T.values
dataPosition = data.copy().T
dataPosition[dataPosition>0] = 1
#==============================================================================
# data = dataPosition.T
#==============================================================================
uNum = len(data.columns)
mNum = len(data.index)
print(uNum, mNum)

point = int(64)
scale = 0.1
lamb = 0.01
method = tf.nn.softplus

#==============================================================================
# #userMatrix
#==============================================================================
x_u = tf.placeholder('float',[None,mNum])
W_u_enco = tf.Variable(tf.random_normal([mNum,point],stddev=0.01))
b_u_enco = tf.Variable(tf.zeros([point]))
y_u_coded = method(tf.add(tf.matmul(x_u,W_u_enco),b_u_enco))
W_u_deco = tf.Variable(tf.zeros([point,mNum]))
b_u_deco = tf.Variable(tf.zeros([mNum]))
y_u_decoded = tf.matmul(y_u_coded,W_u_deco)+b_u_deco
loss_u = tf.reduce_sum(tf.pow(tf.subtract(y_u_decoded,x_u), 2.0)) \
+ tf.nn.l2_loss(W_u_enco) + tf.nn.l2_loss(b_u_enco) \
+ tf.nn.l2_loss(W_u_deco) + tf.nn.l2_loss(b_u_deco)

#==============================================================================
# #movieMatrix
#==============================================================================
x_m = tf.placeholder('float',[None,uNum])
x_m_noise = tf.add(x_m, scale*tf.random_normal((uNum,)))
#==============================================================================
# W_m_enco = tf.Variable(tf.zeros([uNum,point]))
#==============================================================================
W_m_enco = tf.Variable(tf.random_normal([uNum,point],stddev=0.01))
b_m_enco = tf.Variable(tf.zeros([point]))
y_m_coded = method(tf.add(tf.matmul(x_m,W_m_enco),b_m_enco))
W_m_deco = tf.Variable(tf.zeros([point,uNum]))
b_m_deco = tf.Variable(tf.zeros([uNum]))
y_m_decoded = tf.add(tf.matmul(y_m_coded,W_m_deco), b_m_deco)
loss_m = tf.reduce_sum(tf.pow(tf.subtract(y_m_decoded,x_m), 2.0)) \
+ tf.nn.l2_loss(W_m_enco) + tf.nn.l2_loss(b_m_enco) \
+ tf.nn.l2_loss(W_m_deco) + tf.nn.l2_loss(b_m_deco)

#==============================================================================
# #matrix
#==============================================================================
prediction = tf.matmul(y_u_coded,tf.transpose(y_m_coded))
predictionInTrain = prediction*dataPosition.values
loss_matrix = tf.reduce_sum(tf.pow(tf.subtract(predictionInTrain,x_u),2.0))\
+ lamb*(tf.nn.l2_loss(prediction) \
+ tf.nn.l2_loss(W_u_enco) + tf.nn.l2_loss(b_u_enco) \
+ tf.nn.l2_loss(W_u_deco) + tf.nn.l2_loss(b_u_deco) \
+ tf.nn.l2_loss(W_m_enco) + tf.nn.l2_loss(b_m_enco) \
+ tf.nn.l2_loss(W_m_deco) + tf.nn.l2_loss(b_m_deco))

#==============================================================================
# loss_matrix = tf.reduce_sum(tf.pow(tf.subtract(prediction,x_u),2.0))
#==============================================================================


#combine
loss = 0.5*loss_m + 0.5*loss_u + loss_matrix
train_step_1 = tf.train.AdadeltaOptimizer(1).minimize(loss_matrix)
train_step = tf.train.AdadeltaOptimizer().minimize(loss_matrix)

#run
init = tf.global_variables_initializer()
sess.run(init)
for i in range(300):
	# batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step_1, feed_dict={x_u: data.T,x_m: data})
	print(sess.run(loss_matrix,feed_dict={x_u: data.T,x_m: data}),i)
for i in range(300):
	# batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x_u: data.T,x_m: data})
	print(sess.run(loss_matrix,feed_dict={x_u: data.T,x_m: data}),i)

p = sess.run(prediction,feed_dict={x_u: data.T,x_m: data})
p_in_train = p*dataPosition
num_in_train = dataPosition.values.sum()
avg_loss = abs(p_in_train.values - data.T.values).sum()/num_in_train
y_u_decoded_  = sess.run(y_u_decoded ,feed_dict={x_u: data.T,x_m: data})

y_u_coded_ = sess.run(y_u_coded,feed_dict={x_u: data.T,x_m: data})
b = sess.run(b_m_enco,feed_dict={x_u: data.T,x_m: data})
b_ = sess.run(b_m_deco, feed_dict={x_u: data.T,x_m: data})

test = pd.read_csv('input/ratMatrixTest.csv',index_col=0)
test = test.reindex(index = list(data.index), columns = list(data.columns)).T.fillna(0)
testPosition = test.copy()
testPosition[testPosition>0] = 1
p_test = (p*testPosition)
num_in_test = testPosition.values.sum()
avg_loss_test = abs(p_test.values - test.values).sum() / num_in_test