import numpy as np
import joblib
from typing import Optional, Any
from .cleaners import Cleaner
from .features import FeatureExtractor
from .windowing import Windowing
from core.io.utils import save_json, load_json
import logging

logger = logging.getLogger(__name__)

class Processor:
    _cleaner_filename = 'processor_cleaner.joblib'
    _extractor_filename = 'processor_extractor.joblib'
    _details_filename = 'details_processor.json'

    def __init__(
        self, 
        fs: int,
        window_size: float,
        step_size: float,
        feature_extractor: FeatureExtractor,
        cleaner: Optional[Cleaner] = None,
    ):
        self.window_size, self.step_size, self.fs = window_size, step_size, fs
        self.window_samples = int(self.window_size * self.fs)
        self.step_samples = int(self.step_size * self.fs)
        
        self.cleaner = cleaner or self._create_default_cleaner()
        self.windowing = Windowing(self.window_samples, self.step_samples)
        self.feature_extractor = feature_extractor
    
    @property
    def details(self) -> dict[str, Any]:
        return {
            'fs': self.fs,
            'window_size': self.window_size,
            'step_size': self.step_size,
            'cleaner': str(self.cleaner),
            'feature_extractor': str(self.feature_extractor),
        }

    def _create_default_cleaner(self) -> Cleaner:
        from .cleaners import SequentialCleaner, BandpassCleaner, NotchCleaner

        if self.fs > 1000:
            label = 'EMG'
            low, high = 20, 450
        else:
            label = 'EEG'
            low, high = 1, 40

        cleaner = SequentialCleaner(
            BandpassCleaner(fs=self.fs, low=low, high=high),
            NotchCleaner(fs=self.fs),
        )
        logger.info(f'Using default cleaner with {label} bands')
        return cleaner
    
    def fit_transform(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        groups: np.ndarray,
        ignore_ys: Optional[list[Any]] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = self.cleaner.fit_transform(X, y)
        X = self.windowing.transform(X)
        y = self.windowing.transform_center(y)
        groups = self.windowing.transform_center(groups)
        
        if ignore_ys is not None:
            mask = np.isin(y, ignore_ys)
            X = X[~mask]
            y = y[~mask]
            groups = groups[~mask]
        
        X = self.feature_extractor.fit_transform(X, y)
        return X, y, groups

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self.cleaner.transform(X)
        X = self.windowing.transform(X)
        X = self.feature_extractor.transform(X)
        return X
    
    def save(
        self, 
        dir_path: str,
        cleaner_filename: str = _cleaner_filename,
        extractor_filename: str = _extractor_filename,
        details_filename: str = _details_filename,
    ):
        joblib.dump(self.cleaner, f'{dir_path}/{cleaner_filename}')
        joblib.dump(self.feature_extractor, f'{dir_path}/{extractor_filename}')
        save_json(self.details, f'{dir_path}/{details_filename}')

        logger.info(f'Processor saved to {dir_path}')
    
    @classmethod
    def load(
        cls, 
        dir_path: str,
        cleaner_filename: str = _cleaner_filename,
        extractor_filename: str = _extractor_filename,
        details_filename: str = _details_filename,
    ):
        cleaner = joblib.load(f'{dir_path}/{cleaner_filename}')
        feature_extractor = joblib.load(f'{dir_path}/{extractor_filename}')
        details = load_json(f'{dir_path}/{details_filename}')

        return cls(
            fs=details['fs'],
            window_size=details['window_size'],
            step_size=details['step_size'],
            cleaner=cleaner,
            feature_extractor=feature_extractor
        )