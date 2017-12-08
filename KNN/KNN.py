# -*- coding: utf-8 -*-
"""
Created on ：2017/11/15
@author: Freeman
"""


import numpy as np


class KNN:

    def _normalizing(self, X):                             #   最小值最大值算法进行归一化
        X = X.astype(float)
        m = X.shape[1]
        for i in range(m):
            minVal = X[:, i].min()
            maxVal = X[:, i].max()
            X[:, i] = (X[:, i] - minVal) / float(maxVal - minVal)
        return X

    def clfKNN(self, data, test, k=4):
        x = np.array(data)[:, :-1]
        x = self._normalizing(x)
        y = np.array(data)[:, -1]
        m = len(x)
        diffMat = np.tile(test, (m, 1)) - x                  #  构建 m*1 矩阵，用于计算距离
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances ** 0.5
        sortIndex = distances.argsort()                      #  得到distances从小到大排列，y的Index排序
        classCount = {}
        for i in range(k):
            voteLabel = y[sortIndex[i]]
            weight = 1 / (distances[i] ** 2)                 #  在最邻近的k个做表决时，为了减少k值的影响，
                                                             #  给每个投票一个权值weight = 1/d**2
            classCount[voteLabel] = classCount.get(voteLabel, 0) + weight
        sortedClassCount = sorted(classCount.items(),
                                  key=lambda x:x[1], reverse=True)
        return sortedClassCount[0][0]
