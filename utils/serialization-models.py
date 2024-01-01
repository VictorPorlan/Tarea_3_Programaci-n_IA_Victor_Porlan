from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

import pickle

iris = datasets.load_iris()
X_petalos = iris.data[:, 2:4]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X_petalos, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

with open('../models/scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

regresion = LogisticRegression(random_state=42)
regresion.fit(X_train_std, y_train)

svm = SVC(kernel='linear',C=1.0, random_state=1)
svm.fit(X_train_std, y_train)

tree = DecisionTreeClassifier(criterion='gini',max_depth=4,random_state=1)
tree.fit(X_train_std,y_train)

knn = KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)

with open('../models/model-logistic-regression.pck', 'wb') as regresion_file:
    pickle.dump(regresion, regresion_file)

with open('../models/model-svm.pck', 'wb') as svm_file:
    pickle.dump(svm, svm_file)

with open('../models/model-tree.pck', 'wb') as tree_file:
    pickle.dump(tree, tree_file)

with open('../models/model-knn.pck', 'wb') as knn_file:
    pickle.dump(knn, knn_file)