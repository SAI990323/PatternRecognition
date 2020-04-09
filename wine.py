from sklearn.datasets import load_wine
from sklearn import model_selection
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np


wine = load_wine()
wine_data = wine['data']
wine_target = wine['target']
wine_data = StandardScaler().fit_transform(wine_data)
x_train,x_test,y_train,y_test = model_selection.train_test_split(wine_data, wine_target, random_state=1, train_size=0.6)
clf = svm.SVC(C=0.01, kernel='linear', gamma=5, decision_function_shape='ovr')
clf.fit(x_train,y_train)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
print(clf.score(x_train,y_train))
print(clf.score(x_test,y_test))
print(knn.score(x_train,y_train))
print(knn.score(x_test,y_test))

#KNN不能处理样本维度太高的东西，SVM处理高纬度数据比较优秀
"""
　　怎么选择使用二者呢？

　　1 选择KNN的场景：

　　@ 准确度不需要精益求精。

　　@ 样本不多。

　　@ 样本不能一次性获取。智能随着时间一个个得到。

　　2 选择SVM的场景：

　　@ 需要提高正确率。

　　@ 样本比较多。

　　@ 样本固定，并且不会随着时间变化。
"""
