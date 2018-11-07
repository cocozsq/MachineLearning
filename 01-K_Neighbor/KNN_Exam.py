import KNN
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from module import model_selection

'''
raw_data_X =[[3.393533211, 2.331273381],
              [3.110073483, 1.781539638],
              [1.343808831, 3.368360954],
              [3.582294042, 4.679179110],
              [2.280362439, 2.866990263],
              [7.423436942, 4.696522875],
              [5.745051997, 3.533989803],
              [9.172168622, 2.511101045],
              [7.792783481, 3.424088941],
              [7.939820817, 0.791637231]
             ]
raw_data_y = [0,0,0,0,0,1,1,1,1,1]

X_train = np.array(raw_data_X)
y_train = np.array(raw_data_y)

x = np.array([8.093607318, 3.365731514])
print(x.shape)
predict_x = x.reshape(1,-1)
print(predict_x.shape)
neigh = KNN.KNeighborsClassifier(k_neighbors=5)
neigh.fit(X_train, y_train)
result = neigh.predict(predict_x)
print(result)
'''
path = "../data/iris.csv"
def load_file(path):
	iris_df = pd.read_csv(path)
	iris_df.columns = ['sepal_len', 'sepal_width', 'petal_len', 'petal_width', 'class']
	y_train = iris_df['class']
	X_train = iris_df.drop(['class'],axis=1)
	return X_train, y_train
# print(load_file(path))

######################################################################################
# 鸢尾花数据集测试
X, y = load_file(path)

# print(X.shape)
# print(y.shape)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=3)
# print(X_train.shape)
# print(X_test.shape)
neigh = KNN.KNeighborsClassifier(k_neighbors=5)
neigh.fit(X_train, y_train)
test_predict = neigh.predict(X_test)

print(neigh.accuracy_score(y_test,test_predict))

#######################################################################################
'''
# 单组数据测试
X_train, y_train = load_file(path)

X_train = np.array(X_train)
y_train = np.array(y_train)

x_test = np.array([4.9,3.0,1.4,0.2])
# print(x_test.shape)
x_test = x_test.reshape(1,-1)
neigh = KNN.KNeighborsClassifier(k_neighbors=5)
neigh.fit(X_train, y_train)
print(neigh.predict(x_test))
'''