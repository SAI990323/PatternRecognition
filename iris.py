from sklearn.datasets import load_iris
from sklearn import model_selection
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

iris=load_iris()
iris_data = np.array(iris['data'])
iris_target = np.array(iris['target'])

x_train,x_test,y_train,y_test = model_selection.train_test_split(iris_data,iris_target,random_state=1,train_size=0.6)
clf = svm.SVC(C=0.8, kernel='linear', gamma=20, decision_function_shape='ovr')
clf.fit(x_train,y_train)
color = []
for i in iris_target:
    if i == 0:
        color.append('r')
    elif i == 1:
        color.append('g')
    else:
        color.append('y')

# plt.scatter(iris_data[:,0],iris_data[:,1], c=color)
# plt.show()
print("训练集准确率:"+ str(clf.score(x_train,y_train) * 100) + "%")
print("测试集准确率:"+ str(clf.score(x_test,y_test) * 100) + "%")
print("总准确率，" + str(clf.score(iris_data, iris_target) * 100) + "%")


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
print(knn.score(x_train, y_train))
print(knn.score(x_test,y_test))

