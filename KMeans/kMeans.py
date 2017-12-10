# -*- coding: utf-8 -*-
"""
Created on ：2017/12/9
@author: Freeman
"""


from numpy import *


class KMeans:


    def __init__(self, k):
        self.k = k                  #  分为K类
        self.centroids = None       #  簇质点矩阵
        self.clusterAssment = None  #  m * 2 矩阵，第一列储存类别，第二列储存欧式距离

    def _distEclud(self, vecA, vecB):
        ''':return 返回A,B矩阵的欧氏距离'''
        return sqrt(sum(power(vecA - vecB, 2)))

    def _randCent(self, data):
        '''
        随机初始化簇质心矩阵
        :return: 簇质心矩阵
        '''
        n = data.shape[1]
        centroids = mat(zeros((self.k, n)))
        for j in range(n):                             #   按列生成质心矩阵，利用rand生成随机矩阵，在该列的range内
            minVal = data[:, j].min()
            maxVal = data[:, j].max()
            rangeVal = float(maxVal - minVal)          #   该列的range
            centroids[:, j] = minVal + rangeVal * random.rand(self.k, 1)
        return centroids

    def fit(self, data):

        # 类型检查
        if isinstance(data, ndarray):
            pass
        else:
            try:
                data = array(data)
            except:
                raise TypeError("numpy.ndarray required for data")

        m = data.shape[0]                           #  样本数量
        self.centroids = self._randCent(data)       #  初始化 簇质点  矩阵
        self.clusterAssment = mat(zeros((m, 2)))    #  初始化 类别-距离 矩阵
        clusterChanged = True                       #  标志变量，True继续迭代，Flase停止

        while clusterChanged:
            clusterChanged = False
            for i in range(m):
                minDist = inf                       #  初始化最小距离为无穷大
                minIndex = -1                       #  初始化最小距离的索引为-1
                for j in range(self.k):             #  每个样本对每个质点求距离，找出每个样本距离最近的质点
                    distTemp = self._distEclud(data[i, :], self.centroids[j, :])
                    if distTemp < minDist:
                        minDist = distTemp
                        minIndex = j
                if self.clusterAssment[i, 0] != minIndex:       #  只要有样本的分类变动，继续迭代
                    clusterChanged = True
                self.clusterAssment[i, :] = minIndex, minDist   #  更新 类别-距离 矩阵

            for i in range(self.k):
                #  self.clusterAssment[:, 0] == i : 判断 类别-距离矩阵中属于i簇的样本，返回的是 1*m 的布尔型矩阵
                #  nonzero()：返回Ture的索引
                #             返回tuple类型，例如array([0, 1, 3, 5], dtype=int64)，所以要用[0]取出索引
                #             将i簇样本矩阵取出
                ptsInClust = data[nonzero(self.clusterAssment[:, 0] == i)[0]]
                self.centroids[i, :] = mean(ptsInClust, axis=0)  #  按列取平均值作为质点
        return self

    def predict(self, X):

        # 类型检查
        if isinstance(X, ndarray):
            pass
        else:
            try:
                X = array(X)
            except:
                raise TypeError("numpy.ndarray required for X")


        if len(X.shape) == 1:                          #  将一维数组转化为二维数组便于统一逻辑
            m = 1
            X = X.reshape(1,X.shape[0])
        else:
            m = X.shape[0]

        pred = zeros((m, 2))                           #  用于存储结果，第一列为类别，第二列为距离

        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(self.k):
                distTemp = self._distEclud(X[i, :], self.centroids[j, :])
                
                if distTemp < minDist:
                    minDist = distTemp
                    minIndex = j
            pred[i, :] = minIndex, minDist

        return pred