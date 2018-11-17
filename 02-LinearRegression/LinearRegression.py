import numpy as np 
import sys
sys.path.append('../')

from module import metrics

class LinearRegression:

# 初始化线性回归模型（系数，截距，）
	def __init__(self):
		self.coef_ = None  #系数
		self.intercept_ = None #截距
		self._theta = None 

	# 正规化方程的方式求解参数
	def fit_normal(self, X_train, y_train):

		"""根据训练数据集X_train,y_train训练LinearRegression"""
		assert X_train.shape[0] == y_train.shape[0],"the size of X_train must be equal to the size of y_train"
		# 对X_train，加上一列 1。行数为X_train大小，列数为1
		X_b = np.column_stack([np.ones((len(X_train), 1)), X_train])

		# 正规方程求解
		self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

		self.intercept_ = self._theta[0]
		self.coef_ = self._theta[1:]

		return self

	def predict(self, X_predict):
		
		"""
		给定待预测数据集X_predict,返回X_predict的结果向量
		"""
		assert self.intercept_ is not None and self.coef_ is not None,"fit before predict"

		X_b = np.column_stack([np.ones((len(X_predict), 1)), X_predict])
		return X_b.dot(self._theta)
	# R2模型评估
	def score(self, y_predict, y_test):
		# y_predict = self.predict(X_test)
		return metrics.r2_score(y_test, y_predict) 

	def __repr__(self):
		return "LinearRegression()"