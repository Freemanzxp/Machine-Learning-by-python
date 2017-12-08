# -*- coding: utf-8 -*-
"""
Created on ：2017/11/22
@author: Freeman
"""


import numpy as np
import random


class LogisticRegression:

    def __init__(self):
        self.W = 0
        self.b = 0

    def sigmoid(self, data):
        return 1.0 / (1 + np.exp(-data))

    def least_square(self, data):
        x = np.mat(np.array(data))                           # 取原数据作为x，转化为mat矩阵
        x[:, -1] = 1                                         # 将x的最后一列y变为1
        y = np.mat(np.array(data)[:,-1]).T                   # 取原数据的最后一列作为y,需要转置为m*1

        xt_x = x.T * x
        if np.linalg.det(xt_x) == 0.0:
            print('this matrix is singular,cannot inverse')  # 奇异矩阵，不存在逆矩阵
            return

        W_mat = xt_x.I * (x.T * y)                                # 最小二成法公式
        self.W = W_mat[:-1]                                       # 除去最后一个为W权重
        self.b = W_mat[-1]                                        # 最后一个为偏置b

        return self

    def gradientDescent(self,data):
        x = np.mat(np.array(data))                   # 取原数据作为x，转化为mat矩阵
        x[:, -1] = 1                                 # 将x的最后一列y变为1
        y = np.mat(np.array(data)[:, -1]).T          # 取原数据的最后一列作为y,需要转置为m*1
        m, n = np.shape(x)
        alpha = 0.01
        maxCycles = 10000
        weights = np.ones((n, 1))
        for i in range(maxCycles):
            h = self.sigmoid(x * weights)
            loss = h - y
            weights = weights - alpha * x.T * loss   # 将NG的∑公式转化为矩阵形式
        self.W = weights[:-1]                        # 除去最后一个为W权重
        self.b = weights[-1]                         # 最后一个为偏置b
        return self

    def stochasticGradientDescent(self, data, numIter=50000):
        x = np.mat(np.array(data))                   # 取原数据作为x，转化为mat矩阵
        x[:, -1] = 1                                 # 将x的最后一列y变为1
        y = np.mat(np.array(data)[:, -1]).T          # 取原数据的最后一列作为y,需要转置为m*1
        m, n = np.shape(x)
        alpha = 0.01
        weights = np.ones((n, 1))
        for i in range(numIter):
            for j in range(m):
                h = self.sigmoid(x[j] * weights)
                loss = (h - y[j])
                weights = weights - alpha * x[j].T * loss   # 将NG的∑公式转化为矩阵形式,每次用一个样本取更新weights
        self.W = weights[:-1]                                       # 除去最后一个为W权重
        self.b = weights[-1]                                        # 最后一个为偏置b
        return self

    def stochasticGradientDescentNice(self, data, numIter=50000):
        x = np.mat(np.array(data))                   # 取原数据作为x，转化为mat矩阵
        x[:, -1] = 1                                 # 将x的最后一列y变为1
        y = np.mat(np.array(data)[:, -1]).T          # 取原数据的最后一列作为y,需要转置为m*1
        m, n = np.shape(x)
        weights = np.ones((n, 1))
        for i in range(numIter):
            dataIndex = list(range(m))                # 创建一个data索引list
            for j in range(m):
                randIndex = random.choice(dataIndex)    # 在list中随机选择一个index
                alpha = 0.5/(1.0 + i + j) + 0.01        # 增加了学习率的迭代衰减性质，减小震荡幅度
                h = self.sigmoid(x[randIndex] * weights)
                loss = (h - y[randIndex])
                weights = weights - alpha * x[randIndex].T * loss   # 将NG的∑公式转化为矩阵形式,每次用一个样本取更新weights
                dataIndex.remove(randIndex)                         # 在索引list中删除已经使用的index
        self.W = weights[:-1]                                       # 除去最后一个为W权重
        self.b = weights[-1]                                        # 最后一个为偏置b
        return self

    def predict(self, test):
        y_pre = self.sigmoid(np.array(test) * self.W + self.b)
        y = []
        for i in range(len(y_pre)):
            if y_pre[i] > 0.5:
                y.append(1)
            else:
                y.append(0)
        return y
