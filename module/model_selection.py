import numpy as np
import scipy.sparse as sp
		
	
"""
	预备知识：
	稀疏矩阵：存储那些零元素数目远远多于非零元素数目，并且非零元素的分布没有规律的矩阵
	Tools:Scipy库
	存储格式：Sparse Matrix Storage Format(COO)
	（1）、Coordinate Format(COO):坐标形式的系数矩阵，（row, col, data），分别保存非零元素的行，列，与值
		不支持元素存取与增删
	(2)、Diagonal Storage Flrmat(DIA)
		如果稀疏矩阵有仅包含非0元素的对角线，则对角存储格式（DIA)可以减少非零元素定位的信息量
		DIA(values, distance):values，对角元素的值，distance,第i个distance是当前第i个对角线和主对角线的距离
	(3)、Compressed Sparse Row Format(CSR)：压缩稀疏行格式(values, columns ,pointB, pointE)
	(4)、Compressed Sparse Column Format(CSC):压缩稀疏列格式(CSC),列压缩，矩阵A的CSC格式和矩阵A转置的CSR是一样的
"""

"""
description:将系数矩阵转换为压缩稀疏矩阵行格式，并将不可迭代对象转换为数组
	indexable:make arrays indexable for cross_valifation and 
	ensures that everything can be indexed by converting sparse
	matrics to csr and converting non-interable objects to arrays
date:2018-11-05
"""
def indexable(*iterables):
	result = []
	for X in iterables:
		if sp.issparse(X):
			result.append(X.tocsr())
		elif hasattr(X,"__getitem__") or hasattr(X, "iloc"):
			result.append(X)
		elif X is None:
			result.append(X)
		else:
			result.append(np.array(X))
	return result

"""
Validatt
"""

"""
description:split train data and test_data according to concrete scale
date:2018-11-03
reference:sklearn source code
"""

'''
parameters:
-----arrays: features and labels,
-----options:train_size, test_size ......

returns:
-----X_train, X_test, y_train, y_test

'''

def train_test_split(*arrays, **options):
	n_arrays = len(arrays)
	if n_arrays == 0:
		raise ValueError("At least one array required as input")
	# 设置划分比例，如果test_size不存在，按默认值default划分0.25
	test_size = options.pop('test_size','default') 

	train_size = options.pop('train_size',None) 

	random_state = options.pop('random_state',None)
	stratify = options.pop('stratify', None)
	shuffle = options.pop('shuffle', True)
	# 用户传入的参数超出可接受的范围
	if options:
		raise TypeError("Invalid parameters passed:%s" %str(options))

	if test_size == 'default':
		test_size = None
		if train_size is not None:
			warnings.warn("version 0.21 before is ok",FutureWarning)
	if test_size is None and train_size is None:
		test_size = 0.25


	arrays = indexable(*arrays)
	# shuffle:随机打乱顺序，
	# stratify:保持split前类的分布，划分前后类别比例不变，碧泉100个数据，A:80，B:20,A:B=4:1，则划分后训练集与测试集A:B=4;1
	# if shuffle is False:
	# 	if stratify is not None:
	# 		raise ValueError("Strtifid train/test split is not implemented for shuffle=False")
	n_samples = len(arrays[0])
	# print(n_samples)
	if random_state == None:
		random_state=3
	# 设置随机种子，打乱索引
	np.random.seed(random_state)
	shuffle_indexes = np.random.permutation(n_samples)
	tsize = int(n_samples * test_size)
	# 按切割比例分割训练集与测试集
	test_indexes = shuffle_indexes[:tsize]
	train_indexes = shuffle_indexes[tsize:]
	x_train = np.array(arrays[0].iloc[train_indexes])
	y_train = np.array(arrays[1].iloc[train_indexes])
	x_test = np.array(arrays[0].iloc[test_indexes])
	y_test = np.array(arrays[1].iloc[test_indexes])

	return x_train,x_test,y_train,y_test
