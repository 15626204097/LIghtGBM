#  基于LightGBM算法的P2P网贷借款者信用分类研究<br>
>针对借款人的信用分类研究是有效降低P2P网贷信用风险的主要方法。本文利用LightGBM算法在数据分类的高准确率能力，使用特征工程对原始数据的特征提取、选择和重新构建，其中主要采用One hot Encoding编码技术对离散化的特征变量进行重新编码，Z-score数据标准化对连续变量的特征变量进行归一化处理，以及对所有特征变量按贡献度重新排序并进行PCA降维，筛选出用于训练和测试的有效特征变量，最后通过10折交叉验证解决样本的不平衡问题和模型参数寻优的问题。仿真实验证明：LightGBM模型同时具有良好的稳定性、较好的拟合能力、较高的分类预测精度。<br>
##  LightGBM的基本原理<br>
 LightGBM(Light Gradient Boosting Machine) 是微软亚洲研宄院于2016年研发出来的一种最新的分类机器学习算法[13]，该算法是一种开源、快速、高效的基于决策树算法的提升框架，其分类性能在目前众多算法中表现最为出色，具有很多普通机器算法不具备的优点。本文首先介绍LightGBM的相关理论基础---集成学习和boosting、Gradient Boosting、决策树、GBDT等。
集成学习是机器学习的一种分支，LightGBM采用的是集成学习中的提升方法（boosting），boosting算法是一种把若干个分类器整合为一个新分类器的方法[14]，通过增加分错样本的权值和减小分对样本的权值，持续不断的对每个分类器进行学习，同时采用性组合的方式将弱分类器训练成强分类器，以此提高分类器的分类性能和准确率。Boosting的数学表达式如下公式（1）所示：<br>
 ![](https://github.com/15626204097/LIghtGBM/blob/master/image/%E6%89%B9%E6%B3%A8%202020-07-06%20192623.jpg)

Gradient Boosting是一个梯度提升框架，属于Boosting方法中的其中一个，其内部还可以嵌入很多改进算法[15]。Gradient Boosting的主要思想是沿着损失函数的梯度下降方向建立新的模型，其中损失函数（loss function）是衡量模型的预测值和实际值的错误程度，也可以说是模型的偏离实际情况的程度，它是一个非负实值函数，其值越小表明模型的误差越小、拟合程度越好。LightGBM属于Gradient Boosting中的一种改进模型。
本文采用的损失函数是Logistic回归函数，Logistic回归并没有求解出似然函数对数形式的最大值，而是把极大化当做一个引导思想，推导出它的风险函数为负的最小化似然函数。从损失函数的角度上来讲，Logistic回归函数就成为了log损失函数。在公式Logistic回归的推导中，假设样本数据服从伯努利分布（0-1），然后计算满足伯努利分布的极大似然估计函数，通常是先取相应的对数形式，最后求导计算出极值点，目的是方便计算极大似然估计，其中log损失函数的标准形式如下公式（2）所示。
![](https://github.com/15626204097/LIghtGBM/blob/master/image/%E6%89%B9%E6%B3%A8%202020-07-06%20192643.jpg)
式中，L代表的是损失函数， P(Y|X)表示在已知X发生概率下，Y发生的概率，
L(Y, P(Y|X))表示在已知的条件概率分布下，找到能使概率P(Y|X)达到的最大值。
<br>LightGBM中的决策树子模型是采用按叶子分裂的方法（leaf-wise）分裂节点，该分裂方法是指每次从当前所有叶子中找到分裂增益最大的节点，依据此节点对其进行再分裂，而其他不满足收益最大化的节点将停止继续分类，如此不断择优循环生长。根据这种生长规则的方式，可以使LightGBM算法更加高效运行，缺点是会出现生长过深的决策树而出现过拟合现象，通常解决的办法在Leaf-wise之上增加一个最大分裂深度的限制，可以有效防止模型出现过拟合问题，并保证模型的运行效率和精度。Leaf-wise的生长模型图如3所示：

<br>决策树本身属于弱分类器并不能依靠自身找到最优分割点，当样本数据具有高度离散化的特征向量时，弱分类器的分类精度会下降，为了提高分类精度LightGBM选择了基于Histogram的决策树算法，利用Histogram算法的正则化作用避免模型产生过拟合现象。Histogram的工作原理是将连续性的浮点特征变量全部离散化，生成N个离散变量值并构造宽度为N的直方图，然后通过遍历算法重新遍历训练样本，根据离散化后的变量值作为索引并统计每个离散值在直方图中的累计统计量，全部遍历完成后，分析所积累的直方图统计量和直方图的离散值找到最优的分割点。如下图4所示依据Histogram的决策树算法，LightGBM根据周围的叶子直方图以较小的计算量得到周围节点的叶子直方图。
![](https://github.com/15626204097/LIghtGBM/blob/master/image/%E6%89%B9%E6%B3%A8%202020-07-06%20192659.jpg)<br>
![](https://github.com/15626204097/LIghtGBM/blob/master/image/%E6%89%B9%E6%B3%A8%202020-07-06%20192717.jpg)<br>
![](https://github.com/15626204097/LIghtGBM/blob/master/image/%E6%89%B9%E6%B3%A8%202020-07-06%20192729.jpg)<br>
![](https://github.com/15626204097/LIghtGBM/blob/master/image/%E6%89%B9%E6%B3%A8%202020-07-06%20192643.jpg)<br>
![](https://github.com/15626204097/LIghtGBM/blob/master/image/%E6%89%B9%E6%B3%A8%202020-07-06%20192842.jpg)<br>
![](https://github.com/15626204097/LIghtGBM/blob/master/image/%E6%89%B9%E6%B3%A8%202020-07-06%20192854.jpg)<br>
![](https://github.com/15626204097/LIghtGBM/blob/master/image/%E6%89%B9%E6%B3%A8%202020-07-06%20192908.jpg)<br>
![](https://github.com/15626204097/LIghtGBM/blob/master/image/%E6%89%B9%E6%B3%A8%202020-07-06%20192926.jpg)<br>
![](https://github.com/15626204097/LIghtGBM/blob/master/image/%E6%89%B9%E6%B3%A8%202020-07-06%20192952.jpg)<br>
![](https://github.com/15626204097/LIghtGBM/blob/master/image/%E6%89%B9%E6%B3%A8%202020-07-06%20193002.jpg)<br>
![](https://github.com/15626204097/LIghtGBM/blob/master/image/%E6%89%B9%E6%B3%A8%202020-07-06%20193012.jpg)<br>

