import numpy as np

class KnnClassifier:
    def __init__(self, k = 3, ord = 2, kernel = "gaussian", weights = "uniform"):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.ord = ord
        self.kernel = kernel
        self.weights = weights
        self.best_accuracy=-1000

    def _metric(self, a,b):
        return np.linalg.norm(a - b, ord=self.ord)
    
    def _kernel(self, distance, h):
        if self.kernel == "gaussian":
            # Гауссово ядро (классическое, без нормировки)
            return np.exp(-0.5 * (distance / h) ** 2)

        elif self.kernel == "epanechnikov":
            u = distance / h
            return 0.75 * (1 - u ** 2) if abs(u) <= 1 else 0.0

        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel}")

    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        n = X.shape[0]
        distances = np.zeros((X.shape[0], n))
        for i in range(X.shape[0]):
            for j in range(n):
                distances[i][j] = self._metric(X[i], self.X_train[j])
        self.distances = distances

        # for i in range(X.shape[0]):
        #     sorted_indices = np.argsort(distances[i])
        sorted_indices = [np.argsort(distances[i]) for i in range(n)]
        
        #find K with LOO
        classes = np.unique(y)
        for k in range(1, min(n, 10)):
            y_pred = []
            for i in range(n):
                # исключаем саму точку (первый элемент — это индекс самой точки)
                neighbor_idx = sorted_indices[i][1:k+1]

                # h = расстояние до самого дальнего из k соседей
                h = distances[i, neighbor_idx[-1]]

                # вычисляем веса через ядро
                weights = np.array([
                    self._kernel(distances[i, j], h)
                    for j in neighbor_idx
                ])

                # голоса по классам
                votes = np.zeros(len(classes))
                for idx, cls in enumerate(classes):
                    votes[idx] = weights[y[neighbor_idx] == cls].sum()

                # выбираем класс с максимальным суммарным весом
                y_pred.append(classes[np.argmax(votes)])

            y_pred = np.array(y_pred)
            accuracy = np.mean(y_pred == y)
            print(f"k={k}, accuracy={accuracy:.3f}")

            # обновляем лучший результат
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.k = k

        print(f"✅ Best k={self.k}, accuracy={self.best_accuracy:.3f}")
    def _weights(self, distances, k_distance, neighbor_idx):
        match self.weights:
            case "uniform":
                return np.array([1 if i<=self.k else 0 for i in range(len(neighbor_idx))])
            case "kernel":
                return np.array([self._kernel(distances[i], k_distance) for i in neighbor_idx])


            
        
        
    def predict(self, X):
        if self.X_train is None or self.y_train is None:
            raise Exception("Model has not been fitted yet.")
        
        y_pred = []
        classes = np.unique(self.y_train)
        
        for x in X:
            distances = np.array([self._metric(x, xi) for xi in self.X_train])[:self.k]
            neighbor_idx = np.argsort(distances)
            h = distances[-1]
            
            weights = self._weights(distances, h, neighbor_idx)
            votes = np.zeros(len(classes))
            
            for idx, cls in enumerate(classes):
                votes[idx] = weights[self.y_train[neighbor_idx] == cls].sum()
            
            y_pred.append(classes[np.argmax(votes)])
        
        return np.array(y_pred)


    



    
