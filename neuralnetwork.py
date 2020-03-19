import numpy as np

class NeuralNetwork():
	"""docstring for NeuralNetwork"""
	def __init__(self):
		#np.random.seed(25)
		self.weights = np.random.rand(4, 1)
		self.bias = 1
		self.epochs = 50
		self.lr = 0.05

	def ativacao(self, y):
		if y < 0:
			return 0
		else:
			return 1

	def fit(self, data_train, target_train):

		for epoch in range(0, self.epochs):
			for data, target in zip(data_train, target_train):

				out = data.dot(self.weights) #multiplicacao de matrizes
				v0 = out + self.bias
				y_pred = self.ativacao(v0)
				erro = target - y_pred
				delta = erro*data
				temp = delta*self.lr
				temp = temp.reshape(4,1)
				self.weights = self.weights + temp
				self.bias = self.bias + self.lr*erro


	def predict(self, data_test):

		result = []
		for data in data_test:
			out = data.dot(self.weights) #multiplicacao de matrizes
			v0 = out + self.bias
			y_pred = self.ativacao(v0)
			result.append(y_pred)

		print(result)
		return result