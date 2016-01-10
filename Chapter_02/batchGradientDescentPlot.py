import matplotlib.pyplot as plt
from BatchGradientDescent import *
import pandas as pd
import numpy as np 

#read data and look at first and last 5 rows
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
df.tail()
df.head()

#since we want to do binary classification, retain only first 100 rows of data that 
#refers to features for Versicolor and Setosa flowers
#set target variable
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

#set feature / predictor variable. For our purposes, we are only
#going to be using two features: petal length and sepal length
X = df.iloc[0:100, [0, 2]].values


fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (8, 4))

#implement first gradient desecent with learning rate = 0.01
gd1 = BatchGradientDescent(eta = 0.01, n_iter = 10).fit(X, y)
ax[0].plot(range(1, len(gd1.cost_) + 1), np.log10(gd1.cost_), marker = 'o')
ax[0].set_xlabel('Number of iterations')
ax[0].set_ylabel('log10(Sum Of Squared Error)')
ax[0].set_ttile('GradientDescent - Learning Rate = 0.01')

#implement second gradient desecent with learning rate = 0.0001
gd2 = BatchGradientDescent(eta = 0.0001, n_iter = 10).fit(X, y)
ax[1].plot(range(1, len(gd2.cost_) + 1), np.log10(gd2.cost_), marker = 'o')
ax[1].set_xlabel('Number of iterations')
ax[1].set_ylabel('log10(Sum Of Squared Error)')
ax[1].set_ttile('GradientDescent - Learning Rate = 0.0001')

plt.show()

###########implementing GradientDescent using standardized values of X#######

#standardize X 
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

#implement Gradient Descent using standardized X
grad_desc = BatchGradientDescent(eta = 0.01, n_iter = 15)
grad_desc.fit(X_std, y)

#plot observations and decision boundary
plot_decision_regions(X_std, y, classifier = grad_desc)
plt.title('Gradient Descent')
plt.xlabel('sepal length(standardized)')
plt.ylabel('petal length(standardized)')
plt.legend(loc = 'upper left')
plt.show()

#plot number of iterations vs cost at every iteration
plt.plot(range(1, len(grad_desc.cost_) + 1), grad_desc.cost_, marker = 'o')
plt.xlabel('Number of iterations')
plt.ylabel('SSE')
plt.show()


