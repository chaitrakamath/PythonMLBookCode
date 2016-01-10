from stochasticGradientDescent import *
import matplotlib.pyplot as plt
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

#standardize X 
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

#train stochastic gradient descent model and plot the decision boundary
stochastic_gd = stochasticGradientDescent(eta = 0.01, n_iter = 15, random_state = 1)
stochastic_gd.fit(X_std, y)
plt_decision_regions(X_std, y, classifier = stochastic_gd)
plt.title('Stochastic Gradient Descent')
plt.xlabel('sepal length(standardized)')
plt.ylabel('petal length(standardized)')
plt.legend(loc = 'upper left')
plt.show()
plt.plot(range(1, len(stochastic_gd.cost_) + 1), stochastic_gd.cost_, 
	marker = 'o')
plt.xlabel ('Number of iterations')
plt.ylabel ('Average Cost')
plt.show()