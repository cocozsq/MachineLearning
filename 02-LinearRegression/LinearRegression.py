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
	# 梯度下降法求解线性回归模型模型参数
	def fit_gd(self, X_train, y_train, eta=0.01, n_iters = 1e4):
		assert X_train.shape[0] == y_train.shape[0], "the size of X_train must be equal to the size of y_train"

		# 损失函数 MSE
		def J(theta, X_b, y):
			try:
				return np.sum((y-X_b.dot(theta))**2) / len(y)
			except:
				return float('inf')
		# 偏导数，
		def dJ(theta, X_b, y):
			# res = np.empty(len(theta))
			# res[0] = np.sum(X_b.dot(theta) - y)
			# for i in range(1, len(theta)):
			# 	res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])
			# return res*2/len(X_b)
			return X_b.T.dot(X_b.dot(theta) - y) * 2./len(X_b)

		# 梯度下降过程
		def gradient_descent(X_b, y, initial_theta, eta, n_iters = 1e4, epsilon=1e-6):

			theta = initial_theta
			cur_iter = 0 # 迭代次数

			while cur_iter < n_iters:
				gradient = dJ(theta, X_b, y)
				last_theta = theta
				theta = theta - eta*gradient
				if abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon:
					break
				cur_iter += 1
			return theta

		X_b = np.column_stack([np.ones((len(X_train), 1)), X_train])
		initial_theta = np.zeros(X_b.shape[1])

		self._theta =gradient_descent(X_b, y_train, initial_theta, eta, n_iters)
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