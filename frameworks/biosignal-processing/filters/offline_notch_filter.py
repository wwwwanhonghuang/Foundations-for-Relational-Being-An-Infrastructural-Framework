

import numpy as np
from scipy import signal as sp_signal
from typing import Optional, Tuple
from filters.offline_filter import OfflineFilter

class OfflineNotchFilter(OfflineFilter):
    """
    Offline notch filter.
    
    Parameters:
    -----------
    notch_freq : float
        Frequency to remove in Hz
    sampling_rate : float
        Sampling rate in Hz
    quality_factor : float
        Quality factor (default: 30)
    """
    
    def __init__(
        self,
        notch_freq: float,
        sampling_rate: float,
        quality_factor: float = 30.0
    ):
        super().__init__()
        
        self.notch_freq = notch_freq
        self.sampling_rate = sampling_rate
        self.quality_factor = quality_factor
        
        self.b, self.a = self._design_filter()
        
    def _design_filter(self) -> Tuple[np.ndarray, np.ndarray]:
        """Design notch filter."""
        nyquist = self.sampling_rate / 2.0
        normal_freq = self.notch_freq / nyquist
        
        b, a = sp_signal.iirnotch(normal_freq, self.quality_factor)
        return b, a
    
    def filter(self, signal: np.ndarray) -> np.ndarray:
        """Filter entire signal."""
        return sp_signal.filtfilt(self.b, self.a, signal)

