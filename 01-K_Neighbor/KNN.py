# -*- coding:utf-8 -*-
'''
1、KNN算法的原理以及实现
1）、算法描述：
------计算待预测数据点到训练数据集中各个点的距离，选取K个最近的点，并计算这些点的类别概率，以此作为决策依据
2)、优点：
------只包含与未知数据相关的数据，局部模型；无需参数估计，无需训练；多分类问题效果更好
3）、缺点：
------对大型训练数据计算量大
4）、算法过程：
	a）、初始化参数 K
	b)、计算预测点到训练所有点的距离
	c)、随机选取K个距离，进入优先队列
	d)、动态维护大小为k的优先队列
	e)、计算优先队列中元素的概率
	d)、计算精确度，更改不同的k，重新训练
'''

'''
from sklearn.neighbors import KNeighborsClassfier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)
print(neigh.predict([[1.1],[1.2]]))
print(neigh.predict_proba)



'''
import numpy as np
from math import sqrt
from collections import Counter
import queue as Q
import heapq

class KNeighborsClassifier:
	def __init__(self, k_neighbors):
		self.k = k_neighbors
		self._X_train = None
		self._y_train = None

	def fit(self, X_train, y_train):
		self._X_train = X_train
		self._y_train = y_train

		return self

	def predict(self, X_predict):
		y_predict = [self._predict(x) for x in X_predict]
		return np.array(y_predict)

	def _predict(self, x):

		distances = [ sqrt(np.sum((x_train - x )**2)) for x_train in self._X_train ]

		dis = np.argsort(distances)

		top_k = [self._y_train[i] for i in dis[:self.k]]
		votes = Counter(top_k)
		return votes.most_common(1)[0][0]
		
	def accuracy_score(self, y_true, y_predict):
		return sum(y_true == y_predict)/len(y_true)

		# k_neighbors = heapq.nsmallest(k,distances)
		# votes = Counter(k_neighbors)
		# voter = votes.most_common(1)[0][0]
		# return votes[]
		# que = Q.PriorityQueue()
		# try:
		# 	for i in range(k):
		# 		que.put(distances[i])
		# 	for i in range(k, len(distances)):
		# 		if(distances.)
		# except Exception:
		# 	print("k is too large")





