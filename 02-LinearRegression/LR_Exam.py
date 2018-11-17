import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets
import pandas as pd

from LinearRegression import LinearRegression

import sys
sys.path.append('../')
from module.model_selection import train_test_split


"""
鸢尾花数据集：
"""
path = "../data/iris.csv"

def load_file0(path):
	iris_df = pd.read_csv(path)
	iris_df.columns = ['sepal_len', 'sepal_width', 'petal_len', 'petal_width', 'class']
	X = iris_df['petal_len']
	# X = X.reshape(len(X),1)

	y = iris_df['petal_width']
	# y = y.reshape(len(y), 1)
	return X, y

def load_file(path):
	iris_df = pd.read_csv(path)
	iris_df.columns =  ['sepal_len', 'sepal_width', 'petal_len', 'petal_width', 'class']
	iris_df['class_b'] = iris_df['class'].map({'Iris-setosa':1, 'Iris-versicolor':2, 'Iris-virginica':3})
	y = iris_df['class_b']
	y_row = iris_df['class']
	# print(y_row)
	label = np.unique(y_row)
	# print(label)
	# y = []
	# # num = 0
	# for item in y_row:
	# 	if item in label:
	# 		y.append(np.where(item in label))
	# y = np.array(y)
	# 		# num = num+1 
	# print(y)
	# print(np.unique(y))	

	X = iris_df.drop(['class_b','class'],axis=1)
	print(type(X), type(y))	
	return X, y

X, y = load_file0(path)

plt.scatter(X, y)
plt.show()

print(X.ndim, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y)

linReg = LinearRegression()

print(type(X_train))

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

linReg.fit_normal(X_train, y_train)

print("特征参数:", linReg.coef_)

print("截距：",linReg.intercept_)

predictor = linReg.predict(X_test)

print("预测结果:", predictor)

print("模型评分:", linReg.score(predictor, y_test))

'''
<class 'numpy.ndarray'>
(112,) (37,) (112,) (37,)
特征参数: [0.41995622]
截距： -0.3813119958537799
预测结果: [2.13842534 0.16463109 0.20662672 1.55048663 0.24862234 0.16463109
 0.24862234 1.92844723 1.42449976 1.34050852 1.76046474 0.29061796
 0.24862234 1.29851289 1.63447787 1.46649538 1.71846912 1.71846912
 0.16463109 2.05443409 1.76046474 0.24862234 0.20662672 1.97044285
 1.25651727 1.25651727 1.508491   1.508491   0.24862234 0.29061796
 1.38250414 0.29061796 0.24862234 0.20662672 2.30640783 1.508491
 1.29851289]
模型评分: 0.929050859710389

'''