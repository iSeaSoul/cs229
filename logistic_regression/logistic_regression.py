# -*- encoding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import time, random

def sigmod(num):
    return 1.0 / (1 + np.exp(-num))

class BatchGradientDescent(object):
    def __init__(self, train_x, train_y, alpha, iter_tm):
        self.train_x = train_x
        self.train_y = train_y
        self.alpha = alpha
        self.iter_tm = iter_tm
    def run(self):
        n_row, n_col = np.shape(self.train_x)
        coeff = np.ones((n_col, 1))
        for T in xrange(self.iter_tm):
            predict_val = sigmod(self.train_x * coeff)
            error = self.train_y - predict_val
            coeff = coeff + self.alpha * self.train_x.T * error
        return coeff

class StochasticGradientDescent(object):
    def __init__(self, train_x, train_y, alpha, iter_tm):
        self.train_x = train_x
        self.train_y = train_y
        self.alpha = alpha
        self.iter_tm = iter_tm
    def run(self):
        n_row, n_col = np.shape(self.train_x)
        coeff = np.ones((n_col, 1))
        for T in xrange(self.iter_tm):
            for i in xrange(n_row):
                predict_val = sigmod(self.train_x[i, : ] * coeff)
                error = self.train_y[i, : ] - predict_val
                coeff = coeff + self.alpha * self.train_x[i, : ].T * error
        return coeff

class SmoothStochasticGradientDescent(object):
    def __init__(self, train_x, train_y, alpha, iter_tm):
        self.train_x = train_x
        self.train_y = train_y
        self.alpha = alpha
        self.iter_tm = iter_tm
    def run(self):
        n_row, n_col = np.shape(self.train_x)
        coeff = np.ones((n_col, 1))
        for T in xrange(self.iter_tm):
            data_index = range(n_row)
            for i in xrange(n_row):
                alpha = 4.0 / (1.0 + T + i) + 0.01
                rand_index = int(random.uniform(0, len(data_index)))
                predict_val = sigmod(self.train_x[data_index[rand_index], : ] * coeff)
                error = self.train_y[data_index[rand_index], : ] - predict_val
                coeff = coeff + self.alpha * self.train_x[data_index[rand_index], : ].T * error
                del data_index[rand_index]
        return coeff

def load_data(file_name):
    x, y = [], []
    for line in open(file_name, 'r').readlines():
        data = map(float, line.strip().split())
        data.insert(0 , 1)
        x.append(data[: -1])
        y.append(data[-1 :])
    return np.mat(x), np.mat(y)

def draw_result(x, y, coeff, fig_name, used_tm):
    print coeff

    plt.clf()
    plt.title('%s. Used time = %lf' % (fig_name, used_tm))

    n_row, n_col = np.shape(x)
    xClassOne, yClassOne = [] , []
    xClassTwo, yClassTwo = [] , []
    for i in xrange(n_row) :
        if y[i] == 0 :
            xClassOne.append(x[i , 1])
            yClassOne.append(x[i , 2])
        else :
            xClassTwo.append(x[i , 1])
            yClassTwo.append(x[i , 2])

    min_x = min(x[: , 1])[0 , 0]
    max_x = max(x[: , 1])[0 , 0]
    classOne = plt.plot(xClassOne , yClassOne , 'or' , label = 'classOne')
    classTwo = plt.plot(xClassTwo , yClassTwo , 'ob' , label = 'classTwo')

    # w0 * 1 + w1 * x1 + w2 * x2 = 0 -> x2 = (-w0 - w1 * x1) / w2
    coeff = coeff.T
    min_y = float (-coeff[0 , 0] - coeff[0 , 1] * min_x) / coeff[0 , 2]
    max_y = float (-coeff[0 , 0] - coeff[0 , 1] * max_x) / coeff[0 , 2]

    border = plt.plot ([min_x , max_x] , [min_y , max_y] , '-g' , label = 'border')
    plt.xlabel('X1')
    plt.ylabel('X2')

    plt.savefig(fig_name + '.png')

def test_logistic_regression():
    train_x, train_y = load_data('test.txt')

    BGD_method = BatchGradientDescent(train_x, train_y, 0.01, 1000)
    begin_tm = time.time()
    coeff = BGD_method.run()
    used_tm = time.time() - begin_tm
    draw_result(train_x, train_y, coeff, 'BatchGradientDescent', used_tm)

    SGD_method = StochasticGradientDescent(train_x, train_y, 0.01, 10)
    begin_tm = time.time()
    coeff = SGD_method.run()
    used_tm = time.time() - begin_tm
    draw_result(train_x, train_y, coeff, 'StochasticGradientDescent', used_tm)

    SSGD_method = SmoothStochasticGradientDescent(train_x, train_y, 0.01, 10)
    begin_tm = time.time()
    coeff = SSGD_method.run()
    used_tm = time.time() - begin_tm
    draw_result(train_x, train_y, coeff, 'SmoothStochasticGradientDescent', used_tm)

if __name__ == '__main__': 
    test_logistic_regression()
