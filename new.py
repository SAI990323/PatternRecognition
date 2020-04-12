from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
vec = CountVectorizer(stop_words="english",decode_error='ignore')

news = fetch_20newsgroups(subset='all')
news_data = news['data']
vec = CountVectorizer(stop_words='english', decode_error='ignore')
train_vec = vec.fit_transform(news_data)
news_target = news['target']
x_train,x_test,y_train,y_test = model_selection.train_test_split(train_vec, news_target, random_state=1, train_size=0.6)
from sklearn.naive_bayes import MultinomialNB
bays = MultinomialNB()
bays.fit(x_train,y_train)
print(bays.score(x_test, y_test))
from sklearn.metrics import classification_report
y = bays.predict(x_test)
print(classification_report(y_test,y,target_names=news['target_names']))
pca = PCA(n_components=100)
train_vec = pca.fit_transform(train_vec)
x_train,x_test,y_train,y_test = model_selection.train_test_split(train_vec, news_target, random_state=1, train_size=0.6)

#这三个类适用的分类场景各不相同，主要根据数据类型来进行模型的选择。一般来说，如果样本特征的分布大部分是连续值，使用GaussianNB会比较好。如果如果样本特征的分大部分是多元离散值，使用MultinomialNB比较合适。而如果样本特征是二元离散值或者很稀疏的多元离散值，应该使用BernoulliNB。
