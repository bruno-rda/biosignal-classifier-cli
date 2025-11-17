import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.base import BaseEstimator, TransformerMixin

class Cleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def __str__(self):
        return f'{self.__class__.__name__}(**{self.__dict__})'

    def __repr__(self):
        return self.__str__()


class BandpassCleaner(Cleaner):
    def __init__(
        self,
        fs: int,
        low: int = 20,
        high: int = 450,
        order: int = 2
    ):
        self.low, self.high, self.fs, self.order = low, high, fs, order
        nyq = 0.5 * fs
        self.b, self.a = butter(order, [low/nyq, high/nyq], btype='band')

    def transform(self, X: np.ndarray) -> np.ndarray:
        return filtfilt(self.b, self.a, X, axis=0)


class NotchCleaner(Cleaner):
    def __init__(
        self,
        fs: int,
        freq: int = 50,
        Q: int = 30
    ):
        self.freq, self.fs, self.Q = freq, fs, Q
        self.b, self.a = iirnotch(self.freq, self.Q, self.fs)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return filtfilt(self.b, self.a, X, axis=0)


class SequentialCleaner(Cleaner):
    def __init__(self, *cleaners: Cleaner):
        self.cleaners = cleaners

    def transform(self, X: np.ndarray) -> np.ndarray:
        for cleaner in self.cleaners:
            X = cleaner.transform(X)
        return X