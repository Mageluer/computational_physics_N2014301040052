import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.colors as mcl
from sklearn import datasets
import SVM1v1 as pc

"""
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
y[:20]=0
y[50:70]=2
X = df.iloc[0:100, [0, 2]].values
"""
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target[:]
"""
pl.scatter(X[:50,0],X[:50,1],color='red',marker='o',label='setosa')
pl.scatter(X[50:100,0],X[50:100,1],color='blue',marker='x',label='versicolor')
pl.xlabel('petal length')
pl.ylabel('sepal length')
pl.legend(loc='upper left')
pl.show()
"""
ppn = pc.SVM(eta = 0.01, n_iter = 10)
ppn.fit(X, y)
"""
pl.plot(range(1, len(ppn.cost_) + 1), ppn.cost_, marker = 'o')
pl.xlabel('Epoches')
pl.ylabel('Number of misclassifications')
pl.show()
"""
def plot_decision_region(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = mcl.ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    pl.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    pl.xlim(xx1.min(), xx1.max())
    pl.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        pl.scatter(x=X[y == cl, 0],y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
plot_decision_region(X, y, classifier=ppn)
pl.xlabel('sepal length [cm]')
pl.ylabel('petal length [cm]')
pl.legend(loc='upper left')
pl.show()
