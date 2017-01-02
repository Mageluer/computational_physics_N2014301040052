import numpy as np
import matplotlib.pyplot as pl
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
weights, params = [], []
for c in np.arange(-5, 5):
    lr = LogisticRegression(C=10**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
weights = np.array(weights)
pl.plot(params, weights[:, 0], label='petal length')
pl.plot(params, weights[:, 1], linestyle='--', label='petal width')
pl.ylabel('weight coefficient')
pl.xlabel('C')
pl.legend(loc='upper left')
pl.xscale('log')
pl.show()
