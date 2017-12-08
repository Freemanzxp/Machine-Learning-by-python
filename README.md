# Machine-Learning-by-python
利用python实现经典的机器学习算法，希望能将主流的算法集成一个库（Freeman）

## KNN库
基于欧氏距离的最邻近算法
优化：
- 各个维度采取最值标准化，以使得各个特征的影响程度均匀；
- 在K个最邻近点中，给每个点一权重，使得投票更公平，K的影响力最小

## linear_model库
# LinearRegression
线性回归：
- 最小二乘法：least_square
- 梯度下降：
  1.批量梯度下降gradientDescent、
  2.随机梯度下降stochasticGradientDescent、
  3.随机梯度改进stochasticGradientDescentNice（加入了random.choice，增加了随机性）
