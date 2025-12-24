
import numpy as np
from scipy import signal as sp_signal
from typing import Optional, Tuple
from filters.realtime_filter import RealtimeFilter

class RealtimeNotchFilter(RealtimeFilter):
    """
    Realtime notch filter for removing specific frequency (e.g., 50/60 Hz powerline).
    
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
        self.zi = sp_signal.lfilter_zi(self.b, self.a)
        
    def _design_filter(self) -> Tuple[np.ndarray, np.ndarray]:
        """Design notch filter."""
        nyquist = self.sampling_rate / 2.0
        normal_freq = self.notch_freq / nyquist
        
        b, a = sp_signal.iirnotch(normal_freq, self.quality_factor)
        return b, a
    
    def next(self, sample: float, **keywords) -> float:
        """Filter a single sample."""
        filtered, self.zi = sp_signal.lfilter(
            self.b, self.a, [sample], zi=self.zi
        )
        return filtered[0]
    
    def reset(self):
        """Reset filter state."""
        self.zi = sp_signal.lfilter_zi(self.b, self.a)


class RealtimeMovingAverageFilter(RealtimeFilter):
    """
    Simple moving average filter for smoothing.
    
    Parameters:
    -----------
    window_size : int
        Size of moving average window
    """
    
    def __init__(self, window_size: int):
        super().__init__()
        
        if window_size < 1:
            raise ValueError("Window size must be at least 1")
        
        self.window_size = window_size
        self.buffer = []
        
    def next(self, sample: float, **keywords) -> float:
        """Filter a single sample."""
        self.buffer.append(sample)
        
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        
        return sum(self.buffer) / len(self.buffer)
    
    def reset(self):
        """Reset filter state."""
        self.buffer = []

