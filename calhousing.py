import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
m,n = housing.data.shape
print("m", m, "n", n)
# housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data]
# print(housing_data_plus_bias.data.shape)


scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m,1)), scaled_housing_data]

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
Y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="Y")
# theta n+1 x 1 vector? (+1 may be because of bias?)
theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - Y
mse = tf.reduce_mean(tf.square(error), name="mse")
#gradients = 2/m * tf.matmul(tf.transpose(X), error)
#gradients = tf.gradients(mse, [theta])[0]
#training_op = tf.assign(theta, theta - learning_rate*gradients)
training_op = tf.train.GradientDescentOptimizer(0.01).minimize(mse)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0 :
            print("Epoch", epoch, "MSE=", mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()
    print(best_theta)

# with tf.Session() as sess:
#     sess.run(init)
#
#     for epoch in range(n_epochs):
#         if epoch % 100 == 0:
#             print("Epoch", epoch, "MSE =", mse.eval())
#         sess.run(training_op)
#
#     best_theta = theta.eval()
#
# print("Best theta:")
# print(best_theta)
