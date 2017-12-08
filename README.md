# Machine-Learning-by-python
利用python实现经典的机器学习算法，希望能将主流的算法集成一个库（Freeman）

# KNN库
基于欧氏距离的最邻近算法
优化：
- 各个维度采取最值标准化，以使得各个特征的影响程度均匀；
- 在K个最邻近点中，给每个点一权重，使得投票更公平，K的影响力最小

# linear_model库
### LinearRegression
提供了predict方法进行预测

线性回归：
- 最小二乘法：least_square
- 梯度下降：1.批量梯度下降gradientDescent、2.随机梯度下降stochasticGradientDescent、3.随机梯度改进stochasticGradientDescentNice（加入了random.choice，增加了随机性）

### LogisticRegression
提供了predict方法进行预测

逻辑回归（对数几率）：
- 最小二乘法：least_square
- 梯度下降：1.批量梯度下降gradientDescent、2.随机梯度下降stochasticGradientDescent、3.随机梯度改进stochasticGradientDescentNice（加入了random.choice，增加了随机性）

# tree_model库
### DecisionTree：包含ID3、C4.5、CART三种算法
- ID3：根据**信息熵增**进行最佳分割点的选择
- C4.5：根据**信息熵增率**进行最佳分割点的选择
- CART：根据**基尼指数**进行最佳分割点的选择
### 使用方法：
- 实例化DecisionTree，可将mode=ID3、C4.5、CART作为参数输入来选择决策树种类
- fit（）          用训练集训练决策树
- predict（）      用训练好的决策树预测测试集
- show（）         画出决策树
- model_save（）   保存模型，利用pickl库
- model_load（）   载入之前训练好的模型
