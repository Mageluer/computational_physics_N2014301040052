import numpy as np
import perceptron_random as pc

class SVM(object):
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        self.cat_ = np.unique(y)
        self.cat_couple_ = []
        self.ppn_ = []
        for i, cat1 in enumerate(self.cat_[:-1]):
            for cat2 in self.cat_[i+1:]:
                test_X = X[(y == cat1) + (y == cat2)]
                test_y = np.where(y[(y == cat1) + (y == cat2)] == cat1, 1, -1)
                ppn = pc.Perceptron(eta=self.eta, n_iter=self.n_iter, shuffle=self.shuffle, random_state=self.random_state)
                ppn.fit(self._standardize(test_X), test_y)
                self.ppn_.append(ppn)
                self.cat_couple_.append((cat1, cat2))


    def _standardize(self, X):
        X_std = np.copy(X)
        for i in range(X.shape[1]):
            X_std[:,i] = (X[:,i] - X[:,i].mean()) / X[:,i].std()
        return X_std

    def _vote(self, X):
        cat_predict_ = []
        for xi in X:
            score_ = np.zeros(self.cat_.shape[0])
            for cat_couple, ppn in zip(self.cat_couple_, self.ppn_):
                if ppn.predict(xi) == 1:
                    score_[self.cat_ == cat_couple[0]] += 1
                else:
                    score_[self.cat_ == cat_couple[1]] += 1
            cat = self.cat_[score_ == score_.max()][0]
            cat_predict_.append(cat)
        return cat_predict_

    def predict(self, X):
        cat_predict_ = np.array(self._vote(self._standardize(X)))
        return cat_predict_
