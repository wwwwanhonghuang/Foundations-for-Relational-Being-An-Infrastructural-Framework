
import numpy as np
from scipy import signal as sp_signal
from typing import Optional, Tuple
from filters.realtime_filter import RealtimeFilter

class RealtimeLowpassFilter(RealtimeFilter):
    """
    Realtime Butterworth lowpass filter.
    
    Parameters:
    -----------
    cutoff : float
        Cutoff frequency in Hz
    sampling_rate : float
        Sampling rate in Hz
    order : int
        Filter order (default: 4)
    """
    
    def __init__(
        self,
        cutoff: float,
        sampling_rate: float,
        order: int = 4
    ):
        super().__init__()
        
        self.cutoff = cutoff
        self.sampling_rate = sampling_rate
        self.order = order
        
        self.b, self.a = self._design_filter()
        self.zi = sp_signal.lfilter_zi(self.b, self.a)
        
    def _design_filter(self) -> Tuple[np.ndarray, np.ndarray]:
        """Design Butterworth lowpass filter."""
        nyquist = self.sampling_rate / 2.0
        normal_cutoff = self.cutoff / nyquist
        normal_cutoff = max(0.001, min(normal_cutoff, 0.999))
        
        b, a = sp_signal.butter(self.order, normal_cutoff, btype='low')
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