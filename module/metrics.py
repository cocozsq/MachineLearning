import numpy as np 
from math import sqrt

def accracy_score(y_true, y_predict):
	return np.sum(y_true == y_predict)/len(y_true)

def mean_squared_error(y_true, y_predict):
	return np.sum((y_true-y_predict)**2)/len(y_true)

def root_mean_squared_error(y_true, y_predict):
	return np.sqrt(mean_squared_error(y_true, y_predict))

def absolute_mean_squared_error(y_true, y_predict):
	return np.sum(np.absolute(y_true - y_predict))/len(y_true)

def r2_score(y_true, y_predict):
	return 1- mean_squared_error(y_true, y_predict)/np.var(y_true)