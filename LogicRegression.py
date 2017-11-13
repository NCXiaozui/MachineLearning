# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

__author__ = 'XYlander'

import numpy as np
import math as ma

#输入的数据, x为列向量形式
def formal_data(x,y):
    x = np.array(x)
    y = np.array(y)
    return x.T, y

def Learn_funtion(w, x, b):
    result = np.dot(w, x) + b
    return result

def sigmoid(w, x, b):
    lf = Learn_funtion(w, x, b)
    return list(map(lambda x: 1.0/(1.0 + ma.exp(-x)), lf))


def Grandient_Descent(w, b, x, y):
    lam = 0.02
    m = x.shape[1]
    w_new = w
    b_new = b
    
    for i in range(200):
        lr = sigmoid(w_new, x, b_new)
        ls = lr - y
        db = 1/m * np.sum(lr)
        dw = 1/m * np.dot(x, ls.T)
        w_old = w_new
        b_old = b_new
        w_new = w_old - lam * dw
        b_new =  b_old - lam * db
        print(i,dw,db)
    return w_new, b_new

#逻辑回归的代码：
def LogicRegression(x_train, y_train, x_test):
    # 初实化w
    w = np.array([0 for i in range(x_train.shape[0])])   
    b = 0
    w,b = Grandient_Descent(w, b, x_train, y_train)
    result = sigmoid(w, x_test, b)
    print(result)
    return result

if __name__ == '__main__':
    f_train = open('D:/Data/train.txt','r')
    f_test = open('D:/Data/test.txt','r')
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for i in f_train:
        i = i.strip().split('   ')
        x_train.append([float(i[0]),float(i[1])])
        y_train.append(float(i[2]))
    for i in f_test:
        i = i.strip().split('   ')
        x_test.append([float(i[0]),float(i[1])])
        y_test.append(float(i[2]))
    x_train, y_train = formal_data(x_train, y_train)
    x_test, y_test = formal_data(x_test, y_test)
    LogicRegression(x_train, y_train, x_test)
