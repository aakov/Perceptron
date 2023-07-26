import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
      self.eta = eta
      self.n_iter = n_iter
      self.random_state = random_state
      # self.errors_ = []

    def fit(self, X, y):
      rgen = np.random.RandomState(self.random_state)
      self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
      self.errors_ = []

      for _ in range(self.n_iter):
        errors = 0
        for xi, target in zip(X, y):
            update = self.eta * (target - self.predict(xi))
            self.w_[1:] += update * xi
            self.w_[0] += update
            errors += int(update != 0.0)
        self.errors_.append(errors)
      return self

    def net_input(self, X):
      return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
      return np.where(self.net_input(X) >= 0.0, 1, -1)


class SLP(object):
    def __init__(self, eta=0.05, n_iter=10, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = 1

    def fit(self, X, y):
        self.perceptrons = []
        self.errors_ = [0,0,0,0,0,0,0,0,0,0]
        for i in range(len(X)):
            self.perceptron = Perceptron(self.eta, self.n_iter, self.random_state)
            self.perceptron.fit(X, y[i])
            self.perceptrons.append(self.perceptron)
            for j in range(10):
                self.errors_[j] += self.perceptron.errors_[j]


    def predict(self, X):
        prediction = np.zeros((10, 10))
        for i in range(len(X)):
            prediction[i] = self.perceptrons[i].predict(X) # zbior jenowymiarowy idzie w zbior dwuwymiarowy
        return prediction

    def misclassified(self, X, y):
        count = 0
        for i in range(len(X)):
            arr = self.predict(X)[i]
            for j in range(10):
                if arr[j] != y[i][j] :
                    count += 1
        return count

    def show(self, X):
        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        for i, ax in enumerate(axes.flat):
            plot = np.reshape(X[i], (7, 5))
            plot = np.where(plot == -1, 0, plot)
            ax.imshow(plot, cmap="Greys")
        plt.show()


def damage(X, percent, seed=1):
    rgen = np.random.RandomState(seed)
    result = np.array(X)
    count = int(X.shape[1] * percent/100)

    for index_example in range(len(X)):
        order = np.sort(rgen.choice(X.shape[1], count, replace=False))
        for index_pixel in order:
            result[index_example][index_pixel] *= -1

    return result



net = SLP()
df = pd.read_csv('letters.data', header=None)
X = df.iloc[[ 1, 2, 4, 6, 7, 13, 16, 18, 19, 24], :35].values
y = np.array([[-1 for _ in range(10)] for _ in range(10)])
# y = [1, -1, -1, -1, -1, -1,]
np.fill_diagonal(y, 1)

# net.show(X)
# net.fit(X, y)
