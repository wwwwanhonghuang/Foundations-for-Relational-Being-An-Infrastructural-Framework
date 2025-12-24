"""
filters/base_filter.py

Base filter classes and implementations for signal processing.
Provides both realtime and offline filtering capabilities.
"""

import numpy as np
from scipy import signal as sp_signal
from typing import Optional, Tuple
from filters.offline_filter import OfflineFilter

class BaseFilter:
    """Base class for filters - pure contract."""
    
    def __init__(self):
        pass
    
    def filter(self, signal):
        """Filter a signal."""
        raise NotImplementedError("Subclasses must implement filter method")


# # ============================================================================
# # Example Usage
# # ============================================================================

# if __name__ == "__main__":
#     # Generate test signal: 10 Hz sine wave + 50 Hz noise + 100 Hz component
#     duration = 2.0
#     sampling_rate = 1000.0
#     t = np.linspace(0, duration, int(duration * sampling_rate))
    
#     signal_clean = np.sin(2 * np.pi * 10 * t)
#     noise_50hz = 0.5 * np.sin(2 * np.pi * 50 * t)
#     noise_100hz = 0.3 * np.sin(2 * np.pi * 100 * t)
#     noise_random = 0.1 * np.random.randn(len(t))
    
#     signal_noisy = signal_clean + noise_50hz + noise_100hz + noise_random
    
#     print("=" * 60)
#     print("REALTIME FILTERING")
#     print("=" * 60)
    
#     # Test realtime bandpass filter (5-30 Hz)
#     rt_filter = RealtimeBandpassFilter(
#         lowcut=5.0,
#         highcut=30.0,
#         sampling_rate=sampling_rate,
#         order=4
#     )
    
#     rt_filtered = []
#     for sample in signal_noisy:
#         filtered_sample = rt_filter.next(sample)
#         rt_filtered.append(filtered_sample)
    
#     rt_filtered = np.array(rt_filtered)
    
#     print(f"Input signal length: {len(signal_noisy)}")
#     print(f"Filtered signal length: {len(rt_filtered)}")
#     print(f"Input RMS: {np.sqrt(np.mean(signal_noisy**2)):.4f}")
#     print(f"Filtered RMS: {np.sqrt(np.mean(rt_filtered**2)):.4f}")
    
#     print("\n" + "=" * 60)
#     print("OFFLINE FILTERING")
#     print("=" * 60)
    
#     # Test offline bandpass filter (same parameters)
#     offline_filter = OfflineBandpassFilter(
#         lowcut=5.0,
#         highcut=30.0,
#         sampling_rate=sampling_rate,
#         order=4
#     )
    
#     offline_filtered = offline_filter.filter(signal_noisy)
    
#     print(f"Input signal length: {len(signal_noisy)}")
#     print(f"Filtered signal length: {len(offline_filtered)}")
#     print(f"Input RMS: {np.sqrt(np.mean(signal_noisy**2)):.4f}")
#     print(f"Filtered RMS: {np.sqrt(np.mean(offline_filtered**2)):.4f}")
    
#     # Compare realtime vs offline
#     print("\n" + "=" * 60)
#     print("COMPARISON (after initial transient)")
#     print("=" * 60)
    
#     # Skip initial samples to avoid transient effects
#     skip = 500
#     diff = np.abs(rt_filtered[skip:] - offline_filtered[skip:])
#     print(f"Max difference: {np.max(diff):.6f}")
#     print(f"Mean difference: {np.mean(diff):.6f}")
#     print(f"Correlation: {np.corrcoef(rt_filtered[skip:], offline_filtered[skip:])[0,1]:.6f}")
    
#     print("\n" + "=" * 60)
#     print("NOTCH FILTER TEST (removing 50 Hz)")
#     print("=" * 60)
    
#     # Test notch filter
#     notch_filter = OfflineNotchFilter(
#         notch_freq=50.0,
#         sampling_rate=sampling_rate,
#         quality_factor=30.0
#     )
    
#     notch_filtered = notch_filter.filter(signal_noisy)
    
#     # Calculate FFT to verify 50 Hz removal
#     fft_before = np.fft.fft(signal_noisy)
#     fft_after = np.fft.fft(notch_filtered)
#     freqs = np.fft.fftfreq(len(signal_noisy), 1/sampling_rate)
    
#     idx_50hz = np.argmin(np.abs(freqs - 50.0))
#     print(f"50 Hz component before filtering: {np.abs(fft_before[idx_50hz]):.2f}")
#     print(f"50 Hz component after filtering: {np.abs(fft_after[idx_50hz]):.2f}")
#     print(f"Attenuation: {20*np.log10(np.abs(fft_after[idx_50hz])/np.abs(fft_before[idx_50hz])):.1f} dB")