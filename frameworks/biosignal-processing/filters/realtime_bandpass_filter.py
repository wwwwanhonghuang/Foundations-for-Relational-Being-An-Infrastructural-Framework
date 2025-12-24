
from filters.realtime_filter import RealtimeFilter
# ============================================================================
# Realtime Filter Implementations
# ============================================================================

class RealtimeBandpassFilter(RealtimeFilter):
    """
    Realtime Butterworth bandpass filter using IIR filtering.
    
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
        
        # Design filter
        self.b, self.a = self._design_filter()
        
        # Initialize filter state
        self.zi = sp_signal.lfilter_zi(self.b, self.a)
        
    def _design_filter(self) -> Tuple[np.ndarray, np.ndarray]:
        """Design Butterworth bandpass filter."""
        nyquist = self.sampling_rate / 2.0
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        
        # Ensure frequencies are in valid range (0, 1)
        low = max(0.001, min(low, 0.999))
        high = max(low + 0.001, min(high, 0.999))
        
        b, a = sp_signal.butter(self.order, [low, high], btype='band')
        return b, a
    
    def next(self, sample: float, **keywords) -> float:
        """
        Filter a single sample.
        
        Parameters:
        -----------
        sample : float
            Input sample value
            
        Returns:
        --------
        filtered_sample : float
            Filtered output
        """
        filtered, self.zi = sp_signal.lfilter(
            self.b, self.a, [sample], zi=self.zi
        )
        return filtered[0]
    
    def reset(self):
        """Reset filter state to initial conditions."""
        self.zi = sp_signal.lfilter_zi(self.b, self.a)