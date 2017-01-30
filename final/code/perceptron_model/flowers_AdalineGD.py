import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.colors as mcl
import AdalineGD as ad

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values
"""
pl.scatter(X[:50,0],X[:50,1],color='red',marker='o',label='setosa')
pl.scatter(X[50:100,0],X[50:100,1],color='blue',marker='x',label='versicolor')
pl.xlabel('petal length')
pl.ylabel('sepal length')
pl.legend(loc='upper left')
pl.show()
"""
def standardize(X):
    X_std = np.copy(X)
    for i in range(X.shape[1]):
        X_std[:,i] = (X[:,i] - X[:,i].mean()) / X[:,i].std()
    return X_std
ada = ad.AdalineGD(n_iter=10, eta=0.01)
ada.fit(standardize(X), y)

pl.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
pl.xlabel('Epochs')
pl.ylabel('Sum-squared-error')
pl.title('Adaline - Learning rate 0.01')
pl.show()

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

plot_decision_region(standardize(X), y, classifier=ada)
pl.xlabel('sepal length [standardized]($cm$)')
pl.ylabel('petal length [standardized]($cm$)')
pl.title('Adaline - Gradient Descent')
pl.legend(loc='upper left')
pl.show()
