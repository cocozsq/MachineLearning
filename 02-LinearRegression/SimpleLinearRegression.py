import numpy as np 

class SimpleLinearRegression:

	# 初始化线性回归模型
	def __init__(self):
		self.a_ = None
		self.b_ = None

	def fit(self, x_train, y_train):
		x_mean = np.mean(x_train)
		y_mean = np.mean(y_train)

		self.a_ =(x_train - x_mean).dot(y_train - y_mean) / (x_train - x_mean).dot(x_train-x_mean)
		self.b_ =y_mean - self.a_ * x_mean

		return self

	def predict(self, x_predict):
		return np.array([self._predict(x) for x in x_predict])

	def _predict(self, x_single):
		return self.a_ * x_single + self.b_

	def score(self, x_test, y_test):
		y_predict = self.predict(x_test)
		return r2_score(y_test, y_predict)

	def __repr__(self):
		return "SimpleLinearRegression()"
