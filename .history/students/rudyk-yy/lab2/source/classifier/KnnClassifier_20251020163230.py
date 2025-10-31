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
        for k in range(1, min(n, 10)):
            y_pred = []
            for i in range(n):
                
                neighbor_idx = sorted_indices[i][1:k+1]  # Exclude the point itself
                h = distances[i][neighbor_idx[-1]]
                if(i==3):
                    print("ALERT ALERT")
                print("h: ")
                print(h)
                print("neighbor_idx: ")
                print(neighbor_idx[-1])
                print(neighbor_idx)
                print(sorted_indices[i][:k+1])
                print(i)
                
                weights = self._weights(distances[i], h, neighbor_idx)

                votes = np.zeros(len(np.unique(y)))
                
                for idx, cls in enumerate(np.unique(y)):
                    votes[idx] = weights[y[neighbor_idx] == cls].sum()
                y_pred.append(np.unique(y)[np.argmax(votes)])
            
            y_pred = np.array(y_pred)
            accuracy = np.sum(y_pred == y) / n
            
            if k == 1 or accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.k = k
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


    



    
