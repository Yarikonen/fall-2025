
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import comb

class Compactness:
    def __init__(self, X, y):
        """
        X : ndarray of shape (L, d)
            Матрица объектов.
        y : ndarray of shape (L,)
            Метки классов.
        """
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.L = len(y)

    def _nearest_neighbors(self, Omega):
        """
        Находит индексы соседей из множества Ω для каждого объекта.
        """
        X_Omega = self.X[Omega]
        distances = cdist(self.X, X_Omega)
        sorted_idx = np.argsort(distances, axis=1)
        return Omega[sorted_idx]  # возвращаем индексы соседей в исходной выборке

    def compactness_profile(self, Omega):
        """
        Вычисляет профиль компактности Π(m, Ω).
        Возвращает массив длины |Ω|.
        """
        neighbors = self._nearest_neighbors(Omega)
        L = self.L
        Pi = np.zeros(len(Omega))

        for m in range(len(Omega)):
            mismatch = self.y != self.y[neighbors[:, m]]
            Pi[m] = np.mean(mismatch)
        return Pi

    def CCV(self, Omega, k=None):
        if k is None:
            k = len(Omega)
        neighbors = self._nearest_neighbors(Omega)[:, :k]
        mism = (self.y[:, None] != self.y[neighbors]).astype(float)
        weights = comb(self.L - 1, self.L - 1 - np.arange(k)) / comb(self.L, self.L)
        return np.mean(mism @ weights)
