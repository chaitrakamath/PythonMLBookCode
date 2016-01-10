class BatchGradientDescent(object):
	"""Gradient Descent Classifier

	Parameters
	-----------
	eta : float
		Learning rate (between 0.0 and 1.0)
	n_iter : int
		Number of passes over training data set (0 to Inf)


	Attributes
	-----------
	w_ : 1d-array
		Weights after fitting
	errors_ : list
		Number of misclassification errors
	"""

	def __init__(self, eta = 0.01, n_iter = 50):
		self.eta = eta
		self.n_iter = n_iter

	def fit(self, X, y):
		"""Fit training data.

		Parameters
		-----------
		X : array of shape = [n_samples, n_features]
			Training vectors where n_samples is number of observations and
			n_features is number of predictors
		y: array of shape = [n_samples]
			Vector of target values

		Returns
		--------
		self: object
		"""

		#initialize all weights to 0
		self.w_ = np.zeros(1 + X.shape[1])
		self.cost_ = []

		for i in range(self.n_iter):
			output = self.net_input(X)
			errors = (y - output)
			self.w_[1:] += self.eta * X.T.dot(errors)
			self.w_[0] += self.eta * errors.sum()
			cost = (errors ** 2).sum()
			self.cost_.append(cost)
		return self

	def net_input(self, X):
		"""Calculate net input"""
		return np.dot(X, self.w_[1:]) + self.w_[0]

	def activation(self, X):
		"""Compute linear activation"""
		return self.net_input(X)

	def predict(self, X):
		"""Return class label after unit step"""
		return np.where(self.activation(X) >= 0.0, 1, -1)

