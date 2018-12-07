import numpy as np

import tensorflow as tf



X = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],dtype=np.float32)
Y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1],dtype=np.float32)

initial_weight = np.ones_like(X,dtype=np.float32) / X.shape[0]


def cal_error(y_true, y_pred):
    error = y_true != y_pred

    e = error / y_true.shape[0]

    return e


def error_fn(e, epsilon=1e-6):

    alpha = np.log((1-e)/(e+epsilon))*0.5

    return alpha

def update_weight(weight,alpha,y_ture,y_predict):

    def tp_fn():
        update_weight = weight*np.exp(-alpha)/np.sum(weight)
        return update_weight

    def tn_fn():
        update_weight = weight * np.exp(alpha) / np.sum(weight)
        return update_weight

    new_weight = np.where(y_ture==y_predict,tp_fn(),tn_fn())

    return new_weight


def loss_fn(y_true,y_pred):
    loss = tf.losses.hinge_loss(y_true,y_predict)
    return loss


lr = 0.0001
iter = 500


X_input = tf.placeholder(dtype=tf.float32,shape=[None],name='input')
Y_input = tf.placeholder(dtype=tf.float32,shape=[None],name='label')

W1 = tf.Variable(initial_value=tf.random_uniform([1]),dtype=tf.float32)
B1 = tf.Variable(initial_value=tf.constant(0.0),dtype=tf.float32)


y_predict = tf.nn.tanh(X_input*W1 + B1)
loss = loss_fn(Y_input,y_predict)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)

train_op = optimizer.minimize(loss)


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(iter):
        index = np.random.choice((X.shape[0] - 1), size=1)
        x = X[index]
        y = Y[index]
        feed_dict = {X_input: x, Y_input: y}
        l,p = sess.run([loss,y_predict],feed_dict=feed_dict)
        print('{} train: loss = {} , predict = {} , true = {}'.format(i,l,p,y))















