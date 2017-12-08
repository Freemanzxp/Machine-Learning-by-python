# -*- coding: utf-8 -*-
"""
Created on ：2017/12/01
@author: Freeman
"""


import numpy as np


class DecisionTree:
    '''
    构建决策树：
    参数：model：ID3、C4.5、CART
    ID3:不能处理数值型数据，若出现了tree节点中未出现的value会报错
    C4.5：
    CART：
    '''

    def __init__(self, mode='ID3'):
        self._tree = None

        if mode == 'ID3' or mode == 'C4.5' or mode == 'CART':
            self._mode = mode
        else:
            raise Exception('mode should be ID3 or C4.5 or CART')

    def _calcShannonEntropy(self, data):
        '''计算一个完整数据集的信息熵'''
        num = data.shape[0]
        labelCounts = {}
        for featVec in data:
            currentLabel = featVec[-1]
            labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
        shannonEntropy = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key]) / num
            shannonEntropy -= prob * np.log2(prob)

        return shannonEntropy

    def _calcGini(self, data):
        '''计算一个完整数据集的基尼指数'''
        num = data.shape[0]
        labelCounts = {}
        for featVec in data:
            currentLabel = featVec[-1]
            labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
        Gini = 1.0
        for key in labelCounts:
            prob = float(labelCounts[key]) / num
            Gini -= prob ** 2

        return Gini

    def _splitDataSet(self, data, axis, value):
        '''将data按照  某个特征==某个取值  分割出一个数据子集'''
        retIndex = []
        featVec = data[:, axis]
        newData = data[:, [i for i in range(data.shape[1]) if i!=axis]]
        for i in range(len(featVec)):
            if featVec[i] == value:
                retIndex.append(i)
        return newData[retIndex,:]

    def _chooseBestFeatureToSplit_ID3(self, data):
        '''
        对data根据熵进行最佳分割：
        变量说明：numFeatures：特征个数
                 baseEntropy：原始数据集的熵
                 newEntropy：按某个特征分割数据集后的熵
                 infoGain：信息增益
                 bestInfoGain：记录最大的信息增益
                 bestFeatureIndex：信息增益最大时，所选择的分割特征的下标
        
        '''
        numFeatures = data.shape[1] - 1
        baseEntropy = self._calcShannonEntropy(data)
        bestInfoGain = 0.0
        bestFeatureIndex = -1
        for i in range(numFeatures):
            featList = data[:, i]
            valueSet = set(featList)
            newEntropy = 0.0
            for value in valueSet:
                subData = self._splitDataSet(data, i, value)
                prob = len(subData) / float(len(data))
                newEntropy += prob * self._calcShannonEntropy(subData)
            infoGain = baseEntropy - newEntropy
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeatureIndex = i
        return bestFeatureIndex

    def _chooseBestFeatureToSplit_C45(self, data):
        '''
        对data根据熵增益率进行最佳分割：
        变量说明：numFeatures：         特征个数
                 baseEntropy：         原始数据集的熵
                 newEntropy：          按某个特征分割数据集后的熵
                 infoGain：            信息增益
                 sumGain:              计算每个feature总的Gain
                 avgerGain:            每次变化的平均值
                 IV：                  C4.5信息增益率的系数
                 gainRatio：           信息增益率
                 bestGainRatio：       最大的信息增益率
                 bestFeatureIndex：    信息增益最大时，所选择的分割特征的下标

        '''
        numFeatures = len(data[0]) - 1
        baseEntropy = self._calcShannonEntropy(data)
        bestGainRatio = 0.0
        bestFeatureIndex = -1
        for i in range(numFeatures):
            featList = [example[i] for example in data]
            valueSet = set(featList)
            newEntropy = 0.0
            sumGain = 0.0
            avgerGain= 0.0
            IV = 0.0
            for value in valueSet:
                subData = self._splitDataSet(data, i, value)
                prob = len(subData) / float(len(data))
                newEntropy += prob * self._calcShannonEntropy(subData)
                IV -= prob * np.log2(prob)
            infoGain = baseEntropy - newEntropy
            sumGain = sumGain + infoGain
            avgerGain = sumGain / (i + 1)
            gainRatio = infoGain / IV
            #  优化后的C4.5算法，既能保证不偏好取值多的特征，也能保证不偏好取值少的特征。
            if gainRatio > bestGainRatio and infoGain > avgerGain:
                bestGainRatio = gainRatio
                bestFeatureIndex = i
        return bestFeatureIndex

    def _chooseBestFeatureToSplit_CART(self, data):
        '''
        对data根据熵进行最佳分割：
        变量说明：numFeatures：特征个数
                 baseGini：原始数据集的基尼指数
                 newGini：按某个特征分割数据集后的基尼指数
                 infoGini：基尼指数增益
                 bestInfoGini：记录最大的基尼指数增益
                 bestFeatureIndex：基尼指数增益最大时，所选择的分割特征的下标

        '''
        numFeatures = data.shape[1] - 1
        baseGini = self._calcGini(data)
        bestInfoGini = 0.0
        bestFeatureIndex = -1
        for i in range(numFeatures):
            featList = data[:, i]
            valueSet = set(featList)
            newGini = 0.0
            for value in valueSet:
                subData = self._splitDataSet(data, i, value)
                prob = len(subData) / float(len(data))
                newGini += prob * self._calcGini(subData)
            infoGini = baseGini - newGini
            if infoGini > bestInfoGini:
                bestInfoGini = infoGini
                bestFeatureIndex = i
        return bestFeatureIndex

    def _majorityCnt(self, labelList):
        '''投票器，列表中的众数'''
        labelCount = {}
        for vote in labelList:
            labelCount[vote] = labelCount.get(vote, 0) + 1
            sortedLabelCount = sorted(labelCount.iterms(), key=lambda x:x[1], reverse=True)
            return sortedLabelCount[0][0]

    def _createTree(self, data, featList):
        '''
        创建决策树
        :param data: 完整数据集
        :param featList: 特征名称tuple
        :return: 字典树
        '''
        labelList = [example[-1] for example in data]
        #  data中的标签类别一致，终止递归，返回此类型
        if labelList.count(labelList[0]) == len(labelList):
            return labelList[0]
        #  data已经没有可划分标签，但还没有能统一标签，使用投票器选择类别
        if len(data[0]) == 1:
            return self._majorityCnt(data)
        #  data还可以继续分割，选择mode，分割
        if self._mode == 'ID3':
            bestFeatIndex = self._chooseBestFeatureToSplit_ID3(data)
        elif self._mode == 'C4.5':
            bestFeatIndex = self._chooseBestFeatureToSplit_C45(data)
        elif self._mode == 'CART':
            bestFeatIndex = self._chooseBestFeatureToSplit_CART(data)

        bestFeatStr = featList[bestFeatIndex]
        featList = list(featList)
        featList.remove(bestFeatStr)
        featList = tuple(featList)
        #  用字典递归存储树，每个根节点是特征名称，叶子节点是标签
        newTree = {bestFeatStr:{}}
        #  最佳特征的数值集合
        featValueList = [example[bestFeatIndex] for example in data]
        valueSet = set(featValueList)
        for value in valueSet:
            newData = self._splitDataSet(data, bestFeatIndex, value)
            newTree[bestFeatStr][value] = self._createTree(newData, featList)
        return newTree

    def fit(self, data):
        if isinstance(data, np.ndarray):
            pass
        else:
            try:
                data = np.array(data)
                print(data)
            except:
                raise TypeError("numpy.ndarray required for data")
        featList = tuple(['x' + str(i) for i in range(len(data[0])-1)])
        self._tree = self._createTree(data, featList)
        return self

    def predict(self, X):
        if self._tree == None:
            raise NotFittedError("Model not fitted, call `fit` first")

        if isinstance(X, np.ndarray):
            pass
        else:
            try:
                X = np.array(X)
            except:
                raise TypeError("numpy.ndarray required for X")

        def _classify(tree, sample):
            """
            用训练好的决策树对输入数据分类 
            决策树的构建是一个递归的过程，用决策树分类也是一个递归的过程
            _classify()一次只能对一个样本（sample）分类
            To Do: 多个sample的预测怎样并行化？
            """
            #  取出树当前层的key：x1,x2,x3....
            featIndex = list(tree.keys())[0]
            secondDict = tree[featIndex]
            key = sample[int((featIndex)[1])]
            valueOfkey = secondDict[key]
            if isinstance(valueOfkey, dict):
                label = _classify(valueOfkey, sample)
            else:
                label = valueOfkey
            return label

        #  一维array，shape=(n,)
        if len(X.shape) == 1:
            return _classify(self._tree, X)
        else:
            #  二维array, shape=(m,n), 拆分成X[i]==>shape=(n,)
            results = []
            for i in range(X.shape[0]):
                results.append(_classify(self._tree, X[i]))
            return np.array(results)

    def show(self):
        if self._tree == None:
            raise NotFittedError("Model not fitted, call `fit` first")

        # plot the tree using matplotlib
        from tree_model import treePlotter
        treePlotter.createPlot(self._tree)

    def model_save(self, filename):
        import pickle
        fw = open(filename, 'wb')
        pickle.dump(self._tree, fw)
        fw.close()

    def model_load(self,filename):
        import pickle
        fr = open(filename, 'rb')
        self._tree = pickle.load(fr)














