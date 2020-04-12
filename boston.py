from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import model_selection
import numpy as np

boston = load_boston()
boston_data = boston['data']
boston_data = StandardScaler().fit_transform(boston_data)
boston_target = boston['target']
name = boston['feature_names']
i_ = []
for i in range(len(boston_target)):
    if boston_target[i] == 50:
        i_.append(i)

boston_target = np.delete(boston_target, i_, axis=0)
boston_data = np.delete(boston_data, i_, axis=0)

PCA = PCA(n_components=5)
boston_data = PCA.fit_transform(boston_data)

x_train,x_test,y_train,y_test = model_selection.train_test_split(boston_data, boston_target, random_state=1, train_size=0.6)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train, y_train)
lr_y_predict=lr.predict(x_test)
from sklearn.metrics import r2_score
score = r2_score(y_test, lr_y_predict)
print(score)
