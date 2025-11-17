import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import CCA
from scipy.fft import rfft, rfftfreq
from scipy.signal import welch
from mne.decoding import CSP

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def __str__(self):
        return f'{self.__class__.__name__}(**{self.__dict__})'

    def __repr__(self):
        return self.__str__()


class PerChannelExtractor(FeatureExtractor):
    """Base class for extractors that process each channel independently"""
    
    def transform(self, X_windows: np.ndarray) -> np.ndarray:
        """
        Args:
            X_windows: (n_windows, window_length, n_channels)
        Returns:
            features: (n_windows, n_features_per_channel * n_channels)
        """
        n_windows, window_length, n_channels = X_windows.shape
        features_list = []
        
        for window in X_windows:  # (window_length, n_channels)
            window_features = []
            
            for ch in range(n_channels):
                channel_data = window[:, ch]  # (window_length,)
                ch_features = self._extract_channel_features(channel_data)
                window_features.extend(ch_features)
            
            features_list.append(window_features)
        
        return np.array(features_list)
    
    def _extract_channel_features(self, channel_data: np.ndarray) -> list:
        """Override this to implement channel-specific feature extraction"""
        raise NotImplementedError


class MultiChannelExtractor(FeatureExtractor):
    """Base class for extractors that use all channels together"""
    
    def transform(self, X_windows: np.ndarray) -> np.ndarray:
        """
        Args:
            X_windows: (n_windows, window_length, n_channels)
        Returns:
            features: (n_windows, n_features)
        """
        features_list = []
        
        for window in X_windows:  # (window_length, n_channels)
            window_features = self._extract_window_features(window)
            features_list.append(window_features)
        
        return np.array(features_list)
    
    def _extract_window_features(self, window: np.ndarray) -> list:
        """Override this to implement multi-channel feature extraction"""
        raise NotImplementedError


class LinearFeatures(PerChannelExtractor):
    def _extract_channel_features(self, channel_data: np.ndarray) -> list:
        rms = np.sqrt(np.mean(channel_data**2))
        mav = np.mean(np.abs(channel_data))
        wl = np.sum(np.abs(np.diff(channel_data)))
        zc = np.sum(np.diff(np.sign(channel_data)) != 0)
        ssc = np.sum(np.diff(np.sign(np.diff(channel_data))) != 0)
        
        return [rms, mav, wl, zc, ssc]


class CCAFeatures(MultiChannelExtractor):
    def __init__(
        self, 
        fs: int,
        target_frequencies: list[int], 
        n_harmonics: int = 2, 
    ):
        self.target_frequencies = target_frequencies
        self.n_harmonics = n_harmonics
        self.fs = fs
        self._references = None
        
    def _get_references(self, window_length: int) -> dict:
        if self._references is None:
            t = np.arange(window_length) / self.fs
            self._references = {}
            for freq in self.target_frequencies:
                harmonics = []
                for h in range(1, self.n_harmonics + 1):
                    harmonics.append(np.sin(2 * np.pi * h * freq * t))
                    harmonics.append(np.cos(2 * np.pi * h * freq * t))
                self._references[freq] = np.column_stack(harmonics)
        return self._references
    
    def _extract_window_features(self, window: np.ndarray) -> list:
        references = self._get_references(window.shape[0])
        window_features = []
        
        for freq in self.target_frequencies:
            Y = references[freq]
            
            cca = CCA(n_components=1)
            cca.fit(window, Y)
            
            X_c, Y_c = cca.transform(window, Y)
            corr = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]
            
            window_features.append(corr)
        
        # Add pairwise differences
        n_freqs = len(window_features)
        for i in range(n_freqs):
            for j in range(i + 1, n_freqs):
                window_features.append(window_features[i] - window_features[j])
        
        return window_features


class FFTFeatures(PerChannelExtractor):
    def __init__(
        self, 
        fs: int,
        target_frequencies: list[int], 
        n_harmonics: int = 2, 
    ):
        self.target_frequencies = target_frequencies
        self.n_harmonics = n_harmonics
        self.fs = fs
        
    def _extract_channel_features(self, channel_data: np.ndarray) -> list:
        fft_result = rfft(channel_data)
        freqs = rfftfreq(len(channel_data), 1 / self.fs)
        
        features = []
        for freq in self.target_frequencies:
            for h in range(1, self.n_harmonics + 1):
                target_freq = freq * h
                idx = np.argmin(np.abs(freqs - target_freq))
                power = np.abs(fft_result[idx])
                features.append(power)
        
        return features


class PSDFeatures(PerChannelExtractor):
    def __init__(
        self, 
        fs: int,
        target_frequencies: list[int], 
        n_harmonics: int = 2, 
        bandwidth: float = 1.0
    ):
        self.target_frequencies = target_frequencies
        self.n_harmonics = n_harmonics
        self.fs = fs
        self.bandwidth = bandwidth
        
    def _extract_channel_features(self, channel_data: np.ndarray) -> list:
        freqs, psd = welch(channel_data, fs=self.fs, nperseg=len(channel_data))
        
        features = []
        for freq in self.target_frequencies:
            for h in range(1, self.n_harmonics + 1):
                target_freq = freq * h
                
                # Find indices within bandwidth
                freq_mask = (freqs >= target_freq - self.bandwidth/2) & \
                           (freqs <= target_freq + self.bandwidth/2)
                
                # Average power in the band
                band_power = np.mean(psd[freq_mask]) if np.any(freq_mask) else 0.0
                features.append(band_power)
        
        return features


class CSPFeatures(MultiChannelExtractor):
    def __init__(self, n_components: int = 4):
        self.n_components = n_components
        self.csp = CSP(n_components=n_components, log=True)
    
    def fit(self, X_windows: np.ndarray, y: np.ndarray):
        X = np.transpose(X_windows, (0, 2, 1))
        self.csp.fit(X, y)
        return self
    
    def transform(self, X_windows: np.ndarray) -> np.ndarray:
        X = np.transpose(X_windows, (0, 2, 1))
        return self.csp.transform(X)


class CombinedFeatures(FeatureExtractor):
    def __init__(self, *extractors: FeatureExtractor):
        self.extractors = extractors
        
    def transform(self, X_windows: np.ndarray) -> np.ndarray:
        feature_arrays = [ext.transform(X_windows) for ext in self.extractors]
        return np.concatenate(feature_arrays, axis=1)