

import numpy as np
from scipy import signal as sp_signal
from typing import Optional, Tuple
from filters.offline_filter import OfflineFilter

# ============================================================================
# Offline Filter Implementations
# ============================================================================

class OfflineBandpassFilter(OfflineFilter):
    """
    Offline Butterworth bandpass filter using zero-phase filtering (filtfilt).
    
    Parameters:
    -----------
    lowcut : float
        Low cutoff frequency in Hz
    highcut : float
        High cutoff frequency in Hz
    sampling_rate : float
        Sampling rate in Hz
    order : int
        Filter order (default: 4)
    """
    
    def __init__(
        self,
        lowcut: float,
        highcut: float,
        sampling_rate: float,
        order: int = 4
    ):
        super().__init__()
        
        self.lowcut = lowcut
        self.highcut = highcut
        self.sampling_rate = sampling_rate
        self.order = order
        
        self.b, self.a = self._design_filter()
        
    def _design_filter(self) -> Tuple[np.ndarray, np.ndarray]:
        """Design Butterworth bandpass filter."""
        nyquist = self.sampling_rate / 2.0
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        
        low = max(0.001, min(low, 0.999))
        high = max(low + 0.001, min(high, 0.999))
        
        b, a = sp_signal.butter(self.order, [low, high], btype='band')
        return b, a
    
    def filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Filter entire signal using zero-phase filtering.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal
            
        Returns:
        --------
        filtered_signal : np.ndarray
            Filtered output
        """
        return sp_signal.filtfilt(self.b, self.a, signal)

