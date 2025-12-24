"""
detectors/erp_detector.py

Event-Related Potential (ERP) Detection Module - Refactored with Decoupled Filters
Filters are injected as dependencies following the RealtimeFilter/OfflineFilter interface.
"""

import numpy as np
from collections import deque
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

# Assuming filters are imported from filters module
# from filters.base_filter import RealtimeFilter, OfflineFilter


@dataclass
class ERPDetection:
    """Data class for ERP detection results."""
    timestamp: int  # Sample index when detection occurred
    amplitude: float  # Peak amplitude
    latency: float  # Latency from event onset (in samples)
    confidence: float  # Detection confidence score (0-1)
    component: str  # ERP component name (e.g., 'P300', 'N200')
    waveform: Optional[np.ndarray] = None  # Optional: the detected waveform


class BaseERPDetector:
    """Base class for ERP detectors - pure contract."""
    
    def __init__(self):
        pass
    
    def detect(self, signal, events=None):
        """Detect ERPs in a signal."""
        raise NotImplementedError("Subclasses must implement detect method")




# if __name__ == "__main__":
#     # Generate synthetic EEG with P300 response
#     duration = 10.0
#     sampling_rate = 250.0
#     n_samples = int(duration * sampling_rate)
#     t = np.linspace(0, duration, n_samples)
    
#     # Background EEG
#     np.random.seed(42)
#     eeg_signal = np.random.randn(n_samples) * 5
#     eeg_signal += 10 * np.sin(2 * np.pi * 10 * t)
    
#     # Add P300 responses
#     event_times = [1.0, 3.0, 5.0, 7.0, 9.0]
#     event_indices = [int(et * sampling_rate) for et in event_times]
    
#     for event_idx in event_indices:
#         p300_time = np.linspace(0, 0.6, int(0.6 * sampling_rate))
#         p300_wave = 15 * np.exp(-((p300_time - 0.3)**2) / (2 * 0.05**2))
        
#         start = event_idx
#         end = min(event_idx + len(p300_wave), n_samples)
#         wave_len = end - start
#         eeg_signal[start:end] += p300_wave[:wave_len]
    
#     print("=" * 60)
#     print("REALTIME ERP DETECTION WITH INJECTED FILTER")
#     print("=" * 60)
    
#     # Create filter instance
#     realtime_filter = MockRealtimeFilter()
    
#     # Create detector with injected filter
#     rt_detector = RealtimeTemplateMatchingERPDetector(
#         filter=realtime_filter,
#         sampling_rate=sampling_rate,
#         epoch_duration=0.8,
#         baseline_duration=0.2,
#         component_windows={'P300': (0.25, 0.45)},
#         threshold=2.0
#     )
    
#     event_flags = np.zeros(n_samples, dtype=bool)
#     for idx in event_indices:
#         event_flags[idx] = True
    
#     all_detections = []
#     for i, (sample, event_flag) in enumerate(zip(eeg_signal, event_flags)):
#         detections = rt_detector.next(sample, event_flag)
#         if detections:
#             all_detections.extend(detections)
#             for det in detections:
#                 print(f"\nDetected {det.component} at sample {det.timestamp}")
#                 print(f"  Amplitude: {det.amplitude:.2f} µV")
#                 print(f"  Latency: {det.latency*1000:.1f} ms")
#                 print(f"  Confidence: {det.confidence:.3f}")
    
#     print(f"\nTotal realtime detections: {len(all_detections)}")
    
#     print("\n" + "=" * 60)
#     print("OFFLINE ERP DETECTION WITH INJECTED FILTER")
#     print("=" * 60)
    
#     # Create filter instance
#     offline_filter = MockOfflineFilter()
    
#     # Create detector with injected filter
#     offline_detector = OfflineERPDetector(
#         filter=offline_filter,
#         sampling_rate=sampling_rate,
#         epoch_duration=0.8,
#         baseline_duration=0.2,
#         component_windows={'P300': (0.25, 0.45)},
#         threshold=2.0
#     )
    
#     results = offline_detector.detect(eeg_signal, np.array(event_indices))
    
#     print(f"\nProcessed {results['n_epochs']} epochs")
#     print(f"Grand average shape: {results['grand_average'].shape}")
#     print(f"\nDetected components in averaged ERP:")
    
#     for det in results['detections']:
#         print(f"\n{det.component}:")
#         print(f"  Amplitude: {det.amplitude:.2f} µV")
#         print(f"  Latency: {det.latency*1000:.1f} ms")
#         print(f"  Confidence: {det.confidence:.3f}")
    
#     print("\n" + "=" * 60)
#     print("DETECTOR WITHOUT FILTER (raw signal)")
#     print("=" * 60)
    
#     # Create detector without filter
#     no_filter_detector = RealtimeTemplateMatchingERPDetector(
#         filter=None,  # No filtering
#         sampling_rate=sampling_rate,
#         epoch_duration=0.8,
#         baseline_duration=0.2,
#         component_windows={'P300': (0.25, 0.45)},
#         threshold=2.0
#     )
    
#     print("Detector created without filter - processing raw signal directly")