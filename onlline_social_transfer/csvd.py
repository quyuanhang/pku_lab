import sys, time
import tensorflow as tf
import pandas as pd
import numpy as np
from scipy import sparse

def print_schedule(begin, i, s_=None):
    if not s_:
        return 0
    if i % 1000 == 0:
        sum_time = '%0.2f' % (time.time() - begin)
        sys.stderr.write(("\r%s %d sum time %s" % (s_, i, sum_time)))
        sys.stderr.flush()


def complete_schedual():
    sys.stderr.write("\n")
    sys.stderr.flush()

train_male = pd.read_csv('input/male_train.csv', header=None).values
train_female = pd.read_csv('input/female_train.csv', header=None).values

male_set = set(train_male[:, 0]) & set(train_female[:, 0])
female_set = set(train_male[:, 1]) & set(train_female[:, 1])

male_index_dict = dict(zip(male_set, range(len(male_set))))
female_index_dict = dict(zip(female_set, range(len(female_set))))

male_matrix = np.zeros([len(male_set), len(female_set)])
for row in train_male:
    male, female, rate = row
    if male in male_index_dict and female in female_index_dict:
        male_matrix[male_index_dict[male], female_index_dict[female]] = rate
male_matrix_position = male_matrix.copy()
male_matrix_position[male_matrix_position>0] = 1


female_matrix = np.zeros([len(male_set), len(female_set)])
for row in train_female:
    male, female, rate = row
    if male in male_index_dict and female in female_index_dict:
        female_matrix[male_index_dict[male], female_index_dict[female]]
female_matrix_position = female_matrix.copy()
female_matrix_position[female_matrix_position>0] = 1


male_num, female_num = len(male_set), len(female_set)
feature_num = 5
optimizer = tf.train.AdadeltaOptimizer(1)


mat_male_feature = tf.Variable(tf.random_normal((male_num, feature_num)) / tf.sqrt(float(feature_num)))
mat_feature_female = tf.Variable(tf.random_normal((feature_num, female_num)) / tf.sqrt(float(feature_num)))
feature_vec_male = tf.Variable(tf.random_normal(shape=[feature_num,]))
mat_feature = tf.diag(feature_vec_male)
feature_vec_female = tf.Variable(tf.random_normal(shape=[feature_num,]))
mat_feature_ = tf.diag(feature_vec_female)

male_prediction = tf.matmul(tf.matmul(mat_male_feature, mat_feature), mat_feature_female)
female_prediction = tf.matmul(tf.matmul(mat_male_feature, mat_feature_), mat_feature_female)

loss = tf.reduce_sum(tf.pow(tf.subtract(male_prediction * male_matrix_position, male_matrix), 2.0)) \
     + tf.reduce_sum(tf.pow(tf.subtract(female_prediction * female_matrix_position, female_matrix), 2.0))


init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)


begin = time.time()
l_loss = 0
i = 0
while True:
    sess.run(optimizer.minimize(loss))
    c_loss = tf.eval(loss)
    if abs(c_loss - l_loss) < 100:
        break
    l_loss = c_loss
    print_schedule(begin, i, ('loss function %0.1f' % c_loss))
    i += 1
complete_schedual()


