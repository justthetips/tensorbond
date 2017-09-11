import tensorflow as tf
import numpy as np
import pandas as pd

from tensor.bondmath import price


def setup_data(n: int = 10000):
    mats = np.random.randint(1, 31, n)
    cpns = (.1 - 0) * np.random.random(n) + 0
    ylds = (.1 - 0) * np.random.random(n) + 0

    d = {'Yield': ylds, 'Maturity': mats, 'Coupon': cpns}
    df = pd.DataFrame(data=d)
    prices = []
    for index, row in df.iterrows():
        p = price(row['Yield'], int(row['Maturity']), row['Coupon'])
        prices.append(p)
    return np.array(prices), df


# model parameters as external flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', .0001, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 100000, 'Number of steps to run trainer.')
flags.DEFINE_integer('display_step', 1000, 'Display logs per step.')


def run_training(train_X, train_Y):
    m = 10000
    n = 3
    X = tf.placeholder(tf.float32, [m, n])
    Y = tf.placeholder(tf.float32, [m, 1])

    # weights
    W = tf.Variable(tf.zeros([n, 1], dtype=np.float32), name="weight")
    b = tf.constant(0, dtype=np.float32, name="bias")

    # linear model
    activation = tf.add(tf.matmul(X, W), b)
    cost = tf.reduce_sum(tf.square(activation - Y)) / (2 * m)
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cost)

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)

        for step in range(FLAGS.max_steps):

            sess.run(optimizer, feed_dict={X: np.asarray(train_X), Y: np.asarray(train_Y)})

            if step % FLAGS.display_step == 0:
                print("Step {}".format(step + 1))
                print("Cost={:.2f}".format(sess.run(cost, feed_dict={X: np.asarray(train_X), Y: np.asarray(train_Y)})))
                print("W={}".format(sess.run(W)))
                print("b={}".format(sess.run(b)))

        print("Optimization Finished!")
        training_cost = sess.run(cost, feed_dict={X: np.asarray(train_X), Y: np.asarray(train_Y)})
        print("Training Cost={}".format(training_cost))
        print("W={}".format(sess.run(W)))
        print("b={}".format(sess.run(b)))

        print("Predict....")
        predict_X = np.array([.03, 10, .03], dtype=np.float32).reshape((1, 3))

        # Do not forget to normalize your features when you make this prediction
        # predict_X = predict_X / np.linalg.norm(predict_X)

        predict_Y = tf.add(tf.matmul(predict_X, W), b)
        print("House price(Y) ={}".format(sess.run(predict_Y)))


prices, data = setup_data()
data = data.as_matrix()
prices = np.matrix(prices).T
run_training(data,prices)