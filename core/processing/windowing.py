import numpy as np

class Windowing:
    def __init__(self, window_samples: int, step_samples: int):
        self.window_samples = window_samples
        self.step_samples = step_samples

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.array([
            X[i:i+self.window_samples] 
            for i in range(0, len(X) - self.window_samples + 1, self.step_samples)
        ])
    
    def transform_center(self, X: np.ndarray) -> np.ndarray:
        half_window = self.window_samples // 2
        
        return np.array([
            X[i + half_window]
            for i in range(0, len(X) - self.window_samples + 1, self.step_samples)
        ])