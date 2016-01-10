import pandas as pd
import numpy as np 
import matplotlib.pylab as plt
from Perceptron import *
#read data and look at first and last 5 rows
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
df.tail()
df.head()

#since we want to do binary classification, retain only first 100 rows of data that 
#refers to features for Versicolor and Setosa flowers
#set target variable
target = df.iloc[0:100, 4].values
target = np.where(target == 'Iris-setosa', -1, 1)

#set feature / predictor variable. For our purposes, we are only
#going to be using two features: petal length and sepal length
X = df.iloc[0:100, [0, 2]].values

#create a plot of features vs predictor
plt.scatter(X[:50, 0], X[:50, 1], 
	color = 'red', marker = 'o', label = 'setosa')
plt.scatter(X[50:, 0], X[50:, 1], 
	color = 'blue', marker = 'x', label = 'versicolor')
plt.xlabel('petal length')
plt.ylabel('petal height')
plt.legend(loc = 'upper left')
plt.show()

ppn = Perceptron(eta = 0.1, n_iter = 10)
ppt.fit(X, target)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker = 'o')
plt.xlabel('Number of passes through training set')
plt.ylabel('Number of misclassifications')
plt.show()


############################Plot decision boundary for classifier#########
from matplotlib.colors import ListedColormap

def plt_decision_regions(X, y, classifier, resolution = 0.02):
	markers = ('s', 'x', 'o', '^', 'v')
	colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])

	#plot decision surface
	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
		np.arange(x2_max, x2_max, resolution))
	z = classifier.predict(np.array([xx1.reval(), xx2.reval()]).T)
	z = z.reshape(xx1.shape)
	plt.contourf(xx1, xx2, z, alpha = 0.4, cmap = cmap)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())

	#plot class samples
	for idx, c1 in enumerate(np.unique(y)):
		plt.scatter(x = X[y == c1, 0], y = X[y == c1, 1], 
			alpha = 0.8, c = cmap(idx), marker = markers[idx], label = c1)

plt_decision_regions(X, target, classifier = ppn)
plt.xlabel('sepal length (cm)')
plt.ylabel('petal length (cm)')
plt.legend(loc = 'upper left')
plt.show()



