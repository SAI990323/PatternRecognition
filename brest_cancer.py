from sklearn.datasets import load_breast_cancer
from sklearn import model_selection
from sklearn import svm
import numpy as np

brest_cancer = load_breast_cancer()
breast_cancer_data = np.array(brest_cancer['data'])
breast_cancer_target = np.array(brest_cancer['target'])
x_train,x_test,y_train,y_test = model_selection.train_test_split(breast_cancer_data, breast_cancer_target, random_state=1, train_size=0.6)
ctf = svm.SVC(C=0.8, kernel='linear', decision_function_shape='ovo')
ctf.fit(x_train, y_train.ravel())
print("训练集准确率:"+ str(ctf.score(x_train,y_train) * 100) + "%")
print("测试集准确率:"+ str(ctf.score(x_test,y_test) * 100) + "%")
print("总准确率:" + str(ctf.score(breast_cancer_data, breast_cancer_target) * 100) + "%")