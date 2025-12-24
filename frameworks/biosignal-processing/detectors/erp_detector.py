"""
filters/base_filter.py

Base filter classes and implementations for signal processing.
Provides both realtime and offline filtering capabilities.
"""

import numpy as np
from scipy import signal as sp_signal
from typing import Optional, Tuple


class BaseFilter:
    """Base class for filters - pure contract."""
    
    def __init__(self):
        pass
    
    def filter(self, signal):
        """Filter a signal."""
        raise NotImplementedError("Subclasses must implement filter method")


class RealtimeFilter(BaseFilter):
    """Base class for realtime filtering."""
    
    def __init__(self):
        super().__init__()
    
    def next(self, sample, **keywords):
        """Process next sample and return filtered output."""
        raise NotImplementedError("Subclasses must implement next method")
    
    def reset(self):
        """Reset the filter state."""
        raise NotImplementedError("Subclasses must implement reset method")


class OfflineFilter(BaseFilter):
    """Base class for offline/batch filtering."""
    
    def __init__(self):
        super().__init__()
    
    def filter(self, signal):
        """Filter entire signal at once."""
        raise NotImplementedError("Subclasses must implement filter method")


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


class RealtimeHighpassFilter(RealtimeFilter):
    """
    Realtime Butterworth highpass filter.
    
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
        """Design Butterworth highpass filter."""
        nyquist = self.sampling_rate / 2.0
        normal_cutoff = self.cutoff / nyquist
        normal_cutoff = max(0.001, min(normal_cutoff, 0.999))
        
        b, a = sp_signal.butter(self.order, normal_cutoff, btype='high')
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


class OfflineLowpassFilter(OfflineFilter):
    """
    Offline Butterworth lowpass filter.
    
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
        
    def _design_filter(self) -> Tuple[np.ndarray, np.ndarray]:
        """Design Butterworth lowpass filter."""
        nyquist = self.sampling_rate / 2.0
        normal_cutoff = self.cutoff / nyquist
        normal_cutoff = max(0.001, min(normal_cutoff, 0.999))
        
        b, a = sp_signal.butter(self.order, normal_cutoff, btype='low')
        return b, a
    
    def filter(self, signal: np.ndarray) -> np.ndarray:
        """Filter entire signal."""
        return sp_signal.filtfilt(self.b, self.a, signal)


class OfflineHighpassFilter(OfflineFilter):
    """
    Offline Butterworth highpass filter.
    
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
        
    def _design_filter(self) -> Tuple[np.ndarray, np.ndarray]:
        """Design Butterworth highpass filter."""
        nyquist = self.sampling_rate / 2.0
        normal_cutoff = self.cutoff / nyquist
        normal_cutoff = max(0.001, min(normal_cutoff, 0.999))
        
        b, a = sp_signal.butter(self.order, normal_cutoff, btype='high')
        return b, a
    
    def filter(self, signal: np.ndarray) -> np.ndarray:
        """Filter entire signal."""
        return sp_signal.filtfilt(self.b, self.a, signal)


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


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Generate test signal: 10 Hz sine wave + 50 Hz noise + 100 Hz component
    duration = 2.0
    sampling_rate = 1000.0
    t = np.linspace(0, duration, int(duration * sampling_rate))
    
    signal_clean = np.sin(2 * np.pi * 10 * t)
    noise_50hz = 0.5 * np.sin(2 * np.pi * 50 * t)
    noise_100hz = 0.3 * np.sin(2 * np.pi * 100 * t)
    noise_random = 0.1 * np.random.randn(len(t))
    
    signal_noisy = signal_clean + noise_50hz + noise_100hz + noise_random
    
    print("=" * 60)
    print("REALTIME FILTERING")
    print("=" * 60)
    
    # Test realtime bandpass filter (5-30 Hz)
    rt_filter = RealtimeBandpassFilter(
        lowcut=5.0,
        highcut=30.0,
        sampling_rate=sampling_rate,
        order=4
    )
    
    rt_filtered = []
    for sample in signal_noisy:
        filtered_sample = rt_filter.next(sample)
        rt_filtered.append(filtered_sample)
    
    rt_filtered = np.array(rt_filtered)
    
    print(f"Input signal length: {len(signal_noisy)}")
    print(f"Filtered signal length: {len(rt_filtered)}")
    print(f"Input RMS: {np.sqrt(np.mean(signal_noisy**2)):.4f}")
    print(f"Filtered RMS: {np.sqrt(np.mean(rt_filtered**2)):.4f}")
    
    print("\n" + "=" * 60)
    print("OFFLINE FILTERING")
    print("=" * 60)
    
    # Test offline bandpass filter (same parameters)
    offline_filter = OfflineBandpassFilter(
        lowcut=5.0,
        highcut=30.0,
        sampling_rate=sampling_rate,
        order=4
    )
    
    offline_filtered = offline_filter.filter(signal_noisy)
    
    print(f"Input signal length: {len(signal_noisy)}")
    print(f"Filtered signal length: {len(offline_filtered)}")
    print(f"Input RMS: {np.sqrt(np.mean(signal_noisy**2)):.4f}")
    print(f"Filtered RMS: {np.sqrt(np.mean(offline_filtered**2)):.4f}")
    
    # Compare realtime vs offline
    print("\n" + "=" * 60)
    print("COMPARISON (after initial transient)")
    print("=" * 60)
    
    # Skip initial samples to avoid transient effects
    skip = 500
    diff = np.abs(rt_filtered[skip:] - offline_filtered[skip:])
    print(f"Max difference: {np.max(diff):.6f}")
    print(f"Mean difference: {np.mean(diff):.6f}")
    print(f"Correlation: {np.corrcoef(rt_filtered[skip:], offline_filtered[skip:])[0,1]:.6f}")
    
    print("\n" + "=" * 60)
    print("NOTCH FILTER TEST (removing 50 Hz)")
    print("=" * 60)
    
    # Test notch filter
    notch_filter = OfflineNotchFilter(
        notch_freq=50.0,
        sampling_rate=sampling_rate,
        quality_factor=30.0
    )
    
    notch_filtered = notch_filter.filter(signal_noisy)
    
    # Calculate FFT to verify 50 Hz removal
    fft_before = np.fft.fft(signal_noisy)
    fft_after = np.fft.fft(notch_filtered)
    freqs = np.fft.fftfreq(len(signal_noisy), 1/sampling_rate)
    
    idx_50hz = np.argmin(np.abs(freqs - 50.0))
    print(f"50 Hz component before filtering: {np.abs(fft_before[idx_50hz]):.2f}")
    print(f"50 Hz component after filtering: {np.abs(fft_after[idx_50hz]):.2f}")
    print(f"Attenuation: {20*np.log10(np.abs(fft_after[idx_50hz])/np.abs(fft_before[idx_50hz])):.1f} dB")