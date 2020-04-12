
from sklearn.datasets import load_digits
from sklearn import model_selection
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

import numpy as np

from sklearn.decomposition import PCA

id = []
lin_score = []
clf_score = []
knn_score = []
pca_score = []
bayes_score = []
for i in range(1,64):
    digit = load_digits()
    digit_data = digit['data']
    digit_target = digit['target']
    pca = PCA(n_components=i)

    digit_data = pca.fit_transform(digit_data)
    x_train,x_test,y_train,y_test = model_selection.train_test_split(digit_data, digit_target, random_state=1, train_size=0.8)
    svmm = svm.SVC(C=5, kernel='linear', decision_function_shape='ovr')
    svmm.fit(x_train, y_train)
    clf = svm.SVC(C=10, kernel='rbf', decision_function_shape='ovr')
    clf.fit(x_train,y_train)
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(x_train,y_train)
    bays = GaussianNB()
    bays.fit(x_train,y_train)
    bayes_score.append(bays.score(x_test, y_test))
    import matplotlib.pyplot as plt
    lin_score.append(svmm.score(x_test, y_test))
    clf_score.append(clf.score(x_test,y_test))
    knn_score.append(knn.score(x_test,y_test))
    pca_score.append(np.sum(pca.explained_variance_ratio_))
    id.append(i)

plt.plot(id, clf_score, c='red')
plt.plot(id, knn_score, c='blue')
plt.plot(id, lin_score, c='green')
plt.plot(id, pca_score, c='black')
plt.plot(id, bayes_score, c='yellow')
print(clf_score[-1])
print(knn_score[-1])
print(lin_score[-1])
plt.show()
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
