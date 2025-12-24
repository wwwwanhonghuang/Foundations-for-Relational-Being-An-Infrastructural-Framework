

import numpy as np
from collections import deque
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

from decorators.erp.erp_detectpr import RealtimeERPDetector


class RealtimeTemplateMatchingERPDetector(RealtimeERPDetector):
    """
    Realtime ERP detector using template matching and threshold detection.
    
    This detector receives pre-filtered signal through an injected RealtimeFilter.
    
    Parameters:
    -----------
    filter : RealtimeFilter or None
        Filter instance for signal preprocessing. If None, no filtering is applied.
    sampling_rate : float
        Sampling rate in Hz
    epoch_duration : float
        Duration of epoch after event in seconds (e.g., 0.8 for 800ms)
    baseline_duration : float
        Duration of baseline before event in seconds (e.g., 0.2 for 200ms)
    component_windows : dict
        Time windows for ERP components, e.g., {'P300': (0.25, 0.45)}
    threshold : float
        Detection threshold (in standard deviations or absolute amplitude)
    template : Optional[np.ndarray]
        Optional template waveform for matching
    """
    
    def __init__(
        self,
        filter=None,  # RealtimeFilter instance
        sampling_rate: float = 250.0,
        epoch_duration: float = 0.8,
        baseline_duration: float = 0.2,
        component_windows: Optional[Dict[str, Tuple[float, float]]] = None,
        threshold: float = 2.5,
        template: Optional[np.ndarray] = None
    ):
        super().__init__()
        
        # Inject filter dependency
        self.filter = filter
        
        self.sampling_rate = sampling_rate
        self.epoch_duration = epoch_duration
        self.baseline_duration = baseline_duration
        self.threshold = threshold
        self.template = template
        
        # Default component windows (in seconds from event onset)
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
        
        # Calculate buffer sizes
        self.epoch_samples = int(epoch_duration * sampling_rate)
        self.baseline_samples = int(baseline_duration * sampling_rate)
        self.total_buffer_size = self.epoch_samples + self.baseline_samples
        
        # Initialize buffers
        self.signal_buffer = deque(maxlen=self.total_buffer_size)
        self.event_queue = deque()  # Track pending events
        self.sample_counter = 0
        
        # Statistics for adaptive thresholding
        self.baseline_mean = 0.0
        self.baseline_std = 1.0
        
    def _baseline_correct(self, epoch: np.ndarray) -> np.ndarray:
        """Apply baseline correction to epoch."""
        baseline = epoch[:self.baseline_samples]
        baseline_mean = np.mean(baseline)
        return epoch - baseline_mean
    
    def _detect_component(
        self, 
        epoch: np.ndarray, 
        component_name: str, 
        time_window: Tuple[float, float]
    ) -> Optional[Tuple[float, float, int]]:
        """
        Detect a specific ERP component in an epoch.
        
        Returns:
        --------
        (amplitude, latency, peak_idx) or None if not detected
        """
        # Convert time window to sample indices
        start_idx = int(time_window[0] * self.sampling_rate) + self.baseline_samples
        end_idx = int(time_window[1] * self.sampling_rate) + self.baseline_samples
        
        # Ensure indices are within bounds
        start_idx = max(0, min(start_idx, len(epoch) - 1))
        end_idx = max(start_idx + 1, min(end_idx, len(epoch)))
        
        # Extract window
        window = epoch[start_idx:end_idx]
        
        if len(window) == 0:
            return None
        
        # Find peak (positive for P components, negative for N components)
        if component_name.startswith('P'):
            peak_idx_local = np.argmax(window)
            amplitude = window[peak_idx_local]
            threshold_check = amplitude > self.threshold * self.baseline_std
        else:  # N components
            peak_idx_local = np.argmin(window)
            amplitude = window[peak_idx_local]
            threshold_check = amplitude < -self.threshold * self.baseline_std
        
        # Check if peak exceeds threshold
        if threshold_check:
            peak_idx_global = start_idx + peak_idx_local
            latency = (peak_idx_global - self.baseline_samples) / self.sampling_rate
            return amplitude, latency, peak_idx_global
        
        return None
    
    def _match_template(self, epoch: np.ndarray) -> float:
        """Calculate correlation with template if provided."""
        if self.template is None:
            return 0.0
        
        # Ensure same length
        min_len = min(len(epoch), len(self.template))
        epoch_trimmed = epoch[:min_len]
        template_trimmed = self.template[:min_len]
        
        # Normalize and compute correlation
        epoch_norm = (epoch_trimmed - np.mean(epoch_trimmed)) / (np.std(epoch_trimmed) + 1e-10)
        template_norm = (template_trimmed - np.mean(template_trimmed)) / (np.std(template_trimmed) + 1e-10)
        
        correlation = np.correlate(epoch_norm, template_norm, mode='valid')[0] / min_len
        return correlation
    
    def next(
        self, 
        signal_t: float, 
        event_flag: bool = False, 
        **keywords
    ) -> Optional[List[ERPDetection]]:
        """
        Process next signal point and detect ERPs in realtime.
        
        Parameters:
        -----------
        signal_t : float
            Current signal value (raw or pre-filtered)
        event_flag : bool
            True if an event/stimulus occurred at this time
        
        Returns:
        --------
        detections : List[ERPDetection] or None
            List of detected ERPs, or None if no detection
        """
        # Apply filtering if filter is provided
        if self.filter is not None:
            processed_sample = self.filter.next(signal_t)
        else:
            processed_sample = signal_t
        
        # Add to buffer
        self.signal_buffer.append(processed_sample)
        
        # Track events
        if event_flag:
            self.event_queue.append(self.sample_counter)
        
        self.sample_counter += 1
        
        # Check if we need to process any pending events
        detections = []
        
        while self.event_queue:
            event_time = self.event_queue[0]
            samples_since_event = self.sample_counter - event_time
            
            # Check if epoch is complete
            if samples_since_event >= self.epoch_samples:
                # Remove event from queue
                self.event_queue.popleft()
                
                # Check if we have enough history
                if len(self.signal_buffer) >= self.total_buffer_size:
                    # Extract epoch
                    epoch = np.array(list(self.signal_buffer))
                    
                    # Baseline correction
                    epoch_corrected = self._baseline_correct(epoch)
                    
                    # Update baseline statistics
                    baseline = epoch[:self.baseline_samples]
                    self.baseline_mean = np.mean(baseline)
                    self.baseline_std = np.std(baseline) + 1e-10
                    
                    # Detect components
                    for component_name, time_window in self.component_windows.items():
                        result = self._detect_component(
                            epoch_corrected, component_name, time_window
                        )
                        
                        if result is not None:
                            amplitude, latency, peak_idx = result
                            
                            # Calculate confidence
                            confidence = abs(amplitude) / (self.threshold * self.baseline_std)
                            confidence = min(confidence, 1.0)
                            
                            # Add template matching if available
                            if self.template is not None:
                                template_corr = self._match_template(epoch_corrected)
                                confidence = 0.7 * confidence + 0.3 * template_corr
                            
                            detection = ERPDetection(
                                timestamp=event_time,
                                amplitude=amplitude,
                                latency=latency,
                                confidence=confidence,
                                component=component_name,
                                waveform=epoch_corrected.copy()
                            )
                            detections.append(detection)
            else:
                # Not ready yet, stop checking
                break
        
        return detections if detections else None
    
    def reset(self):
        """Reset detector state."""
        self.signal_buffer.clear()
        self.event_queue.clear()
        self.sample_counter = 0
        self.baseline_mean = 0.0
        self.baseline_std = 1.0
        
        # Reset filter if it exists
        if self.filter is not None:
            self.filter.reset()
