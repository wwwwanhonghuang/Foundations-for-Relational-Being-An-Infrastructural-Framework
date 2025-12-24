

import numpy as np
from collections import deque
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

from decorators.erp.erp_detectpr import BaseERPDetector




class MockOfflineFilter:
    """Mock offline filter for demonstration."""
    def filter(self, signal):
        # Simple moving average
        window = 5
        return np.convolve(signal, np.ones(window)/window, mode='same')



class OfflineERPDetector(BaseERPDetector):
    """
    Offline ERP detector for batch processing of recorded data.
    
    Receives pre-filtered signal through an injected OfflineFilter.
    
    Parameters:
    -----------
    filter : OfflineFilter or None
        Filter instance for signal preprocessing. If None, no filtering is applied.
    sampling_rate : float
        Sampling rate in Hz
    epoch_duration : float
        Duration of epoch after event in seconds
    baseline_duration : float
        Duration of baseline before event in seconds
    component_windows : dict
        Time windows for ERP components
    threshold : float
        Detection threshold
    """
    
    def __init__(
        self,
        filter=None,  # OfflineFilter instance
        sampling_rate: float = 250.0,
        epoch_duration: float = 0.8,
        baseline_duration: float = 0.2,
        component_windows: Optional[Dict[str, Tuple[float, float]]] = None,
        threshold: float = 2.5
    ):
        super().__init__()
        
        # Inject filter dependency
        self.filter = filter
        
        self.sampling_rate = sampling_rate
        self.epoch_duration = epoch_duration
        self.baseline_duration = baseline_duration
        self.threshold = threshold
        
        # Default component windows
        if component_windows is None:
            self.component_windows = {
                'N100': (0.08, 0.15),
                'P200': (0.15, 0.25),
                'N200': (0.18, 0.28),
                'P300': (0.25, 0.45),
                'N400': (0.35, 0.55),
            }
        else:
            self.component_windows = component_windows
        
        self.epoch_samples = int(epoch_duration * sampling_rate)
        self.baseline_samples = int(baseline_duration * sampling_rate)
    
    def extract_epochs(
        self, 
        signal: np.ndarray, 
        event_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract epochs around events.
        
        Parameters:
        -----------
        signal : np.ndarray
            Continuous signal (pre-filtered or raw)
        event_indices : np.ndarray
            Indices of event occurrences
            
        Returns:
        --------
        epochs : np.ndarray
            Shape (n_epochs, n_samples)
        valid_events : np.ndarray
            Indices of valid events
        """
        epochs = []
        valid_events = []
        
        for event_idx in event_indices:
            start_idx = event_idx - self.baseline_samples
            end_idx = event_idx + self.epoch_samples
            
            # Check boundaries
            if start_idx >= 0 and end_idx <= len(signal):
                epoch = signal[start_idx:end_idx]
                epochs.append(epoch)
                valid_events.append(event_idx)
        
        return np.array(epochs), np.array(valid_events)
    
    def baseline_correct(self, epochs: np.ndarray) -> np.ndarray:
        """Apply baseline correction to epochs."""
        baseline = epochs[:, :self.baseline_samples]
        baseline_mean = np.mean(baseline, axis=1, keepdims=True)
        return epochs - baseline_mean
    
    def detect(
        self, 
        signal: np.ndarray, 
        event_indices: np.ndarray,
        return_epochs: bool = False
    ) -> Dict:
        """
        Detect ERPs in offline data.
        
        Parameters:
        -----------
        signal : np.ndarray
            Continuous EEG signal (raw)
        event_indices : np.ndarray
            Indices where events occurred
        return_epochs : bool
            If True, return individual epochs
            
        Returns:
        --------
        results : dict
            Dictionary containing detection results
        """
        # Apply filtering if filter is provided
        if self.filter is not None:
            filtered_signal = self.filter.filter(signal)
        else:
            filtered_signal = signal
        
        # Extract epochs
        epochs, valid_events = self.extract_epochs(filtered_signal, event_indices)
        
        if len(epochs) == 0:
            return {'grand_average': None, 'detections': [], 'epochs': None}
        
        # Baseline correction
        epochs_corrected = self.baseline_correct(epochs)
        
        # Compute grand average
        grand_average = np.mean(epochs_corrected, axis=0)
        
        # Time vector
        total_samples = self.baseline_samples + self.epoch_samples
        time_vector = np.linspace(
            -self.baseline_duration, 
            self.epoch_duration, 
            total_samples
        )
        
        # Detect components in grand average
        baseline_std = np.std(grand_average[:self.baseline_samples])
        detections = []
        
        for component_name, time_window in self.component_windows.items():
            start_idx = int(time_window[0] * self.sampling_rate) + self.baseline_samples
            end_idx = int(time_window[1] * self.sampling_rate) + self.baseline_samples
            
            start_idx = max(0, min(start_idx, len(grand_average) - 1))
            end_idx = max(start_idx + 1, min(end_idx, len(grand_average)))
            
            window = grand_average[start_idx:end_idx]
            
            if len(window) == 0:
                continue
            
            # Find peak
            if component_name.startswith('P'):
                peak_idx_local = np.argmax(window)
                amplitude = window[peak_idx_local]
                threshold_check = amplitude > self.threshold * baseline_std
            else:
                peak_idx_local = np.argmin(window)
                amplitude = window[peak_idx_local]
                threshold_check = amplitude < -self.threshold * baseline_std
            
            if threshold_check:
                peak_idx_global = start_idx + peak_idx_local
                latency = (peak_idx_global - self.baseline_samples) / self.sampling_rate
                
                detection = ERPDetection(
                    timestamp=0,  # Not applicable for averaged data
                    amplitude=amplitude,
                    latency=latency,
                    confidence=abs(amplitude) / (self.threshold * baseline_std),
                    component=component_name
                )
                detections.append(detection)
        
        results = {
            'grand_average': grand_average,
            'detections': detections,
            'time_vector': time_vector,
            'n_epochs': len(epochs),
            'valid_event_indices': valid_events
        }
        
        if return_epochs:
            results['epochs'] = epochs_corrected
        
        return results
