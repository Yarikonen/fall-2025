from loss.LogLoss import LogLoss
from optimizer.momentum import Momentum
import numpy as np



class SGDClassifier:
    def __init__(self, learning_rate=0.01, n_iterations=1000, optimizer=Momentum(), loss=LogLoss(), weight_init='random'):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.optimizer = optimizer
        self.loss = loss
        self.bias = None
        
    def _initialize_weights(self, n_features):
        if self.weight_init == 'random':
            self.weights = np.random.randn(n_features)
        elif self.weight_init == 'zeros':
            self.weights = np.zeros(n_features)
        else:
            raise ValueError("weight_init must be 'random' or 'zeros'")
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._initialize_weights(n_features)
        self.bias = 0

        shuffle_indices = np.random.permutation(n_samples)
        X, y = X[shuffle_indices], y[shuffle_indices]
        self.Q = self.loss.loss(y * (np.dot(X, self.weights) + self.bias)).mean()

        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = np.where(linear_model >= 0, 1, -1)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, self.loss.derivative(y * linear_model))
            db = (1 / n_samples) * np.sum(self.loss.derivative(y * linear_model))

            # Update weights and bias using the optimizer
            self.weights = self.optimizer.update(self.weights, lambda w: dw, self.learning_rate)
            self.bias -= self.learning_rate * db
