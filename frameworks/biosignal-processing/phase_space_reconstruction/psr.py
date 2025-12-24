import numpy as np
from collections import deque
from typing import Optional, Tuple

class BasePhaseSpaceReconstructionModel:
    """Base class for phase space reconstruction models - pure contract."""
    
    def __init__(self):
        pass
        
    def reconstruct(self, signal):
        """Reconstruct phase space from a signal."""
        raise NotImplementedError("Subclasses must implement reconstruct method")


class RealtimePhaseSpaceReconstructionModel(BasePhaseSpaceReconstructionModel):
    """Base class for realtime phase space reconstruction."""
    
    def __init__(self):
        super().__init__()
        
    def next(self, signal_t, **keywords):
        """Process next signal point and return phase space coordinates."""
        raise NotImplementedError("Subclasses must implement next method")
    
    def reset(self):
        """Reset the model state."""
        raise NotImplementedError("Subclasses must implement reset method")


class RealtimeTimeDelayMethodPSR(RealtimePhaseSpaceReconstructionModel):
    """
    Realtime Phase Space Reconstruction using Time Delay Embedding method.
    
    Uses Takens' theorem to reconstruct the phase space from a scalar time series
    by creating delay coordinates: [x(t), x(t-τ), x(t-2τ), ..., x(t-(m-1)τ)]
    
    Parameters:
    -----------
    embedding_dimension : int
        The embedding dimension (m)
    time_delay : int
        The time delay (τ) in number of samples
    """
    
    def __init__(self, embedding_dimension: int = 3, time_delay: int = 1):
        super().__init__()
        
        if embedding_dimension < 1:
            raise ValueError("Embedding dimension must be at least 1")
        if time_delay < 1:
            raise ValueError("Time delay must be at least 1")
            
        self.embedding_dimension = embedding_dimension
        self.time_delay = time_delay
        
        # Calculate required history length
        self.required_history = (embedding_dimension - 1) * time_delay + 1
        
        # Initialize circular buffer for history
        self.history = deque(maxlen=self.required_history)
        self.is_initialized = False
        
    def next(self, signal_t: float, return_valid: bool = False, **keywords) -> Optional[np.ndarray]:
        """
        Process the next signal point and return phase space coordinates.
        
        Parameters:
        -----------
        signal_t : float
            The current signal value at time t
        return_valid : bool
            If True, returns tuple (coordinates, is_valid)
        **keywords : dict
            Additional keyword arguments (for compatibility)
            
        Returns:
        --------
        coordinates : np.ndarray or None
            Phase space coordinates [x(t), x(t-τ), ..., x(t-(m-1)τ)]
            Returns None if not enough history is available yet
        (coordinates, is_valid) : tuple
            If return_valid=True, returns tuple with coordinates and validity flag
        """
        # Add new signal point to history
        self.history.append(signal_t)
        
        # Check if we have enough history for reconstruction
        if len(self.history) < self.required_history:
            if return_valid:
                return None, False
            return None
        
        # Mark as initialized once we have enough history
        if not self.is_initialized:
            self.is_initialized = True
        
        # Extract delay coordinates
        coordinates = np.array([
            self.history[-(1 + i * self.time_delay)]
            for i in range(self.embedding_dimension)
        ])
        
        if return_valid:
            return coordinates, True
        return coordinates
    
    def reset(self):
        """Reset the model state, clearing all history."""
        self.history.clear()
        self.is_initialized = False
        
    def reconstruct(self, signal: np.ndarray) -> np.ndarray:
        """
        Reconstruct phase space from a complete signal (batch mode).
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal of shape (n_samples,)
            
        Returns:
        --------
        phase_space : np.ndarray
            Phase space coordinates of shape (n_points, embedding_dimension)
            where n_points = n_samples - (m-1)*τ
        """
        if len(signal) < self.required_history:
            raise ValueError(
                f"Signal length ({len(signal)}) must be at least "
                f"{self.required_history} for m={self.embedding_dimension}, τ={self.time_delay}"
            )
        
        n_points = len(signal) - (self.embedding_dimension - 1) * self.time_delay
        phase_space = np.zeros((n_points, self.embedding_dimension))
        
        for i in range(n_points):
            for j in range(self.embedding_dimension):
                phase_space[i, j] = signal[i + j * self.time_delay]
                
        return phase_space
    
    def get_state(self) -> dict:
        """Get current state of the model."""
        return {
            'embedding_dimension': self.embedding_dimension,
            'time_delay': self.time_delay,
            'history': list(self.history),
            'is_initialized': self.is_initialized
        }
    
    def set_state(self, state: dict):
        """Restore model state."""
        self.embedding_dimension = state['embedding_dimension']
        self.time_delay = state['time_delay']
        self.required_history = (self.embedding_dimension - 1) * self.time_delay + 1
        self.history = deque(state['history'], maxlen=self.required_history)
        self.is_initialized = state['is_initialized']




# # Example usage
# if __name__ == "__main__":
#     # Create a simple sinusoidal signal
#     t = np.linspace(0, 4 * np.pi, 1000)
#     signal = np.sin(t) + 0.5 * np.sin(2 * t)
    
#     # Initialize realtime reconstructor
#     psr = RealtimeTimeDelayMethodPSR(embedding_dimension=3, time_delay=10)
    
#     print("Processing signal in realtime mode...")
#     coordinates_list = []
    
#     for i, s in enumerate(signal):
#         coord = psr.next(s)
#         if coord is not None:
#             coordinates_list.append(coord)
#             if i < 35:  # Print first few valid coordinates
#                 print(f"t={i}: {coord}")
    
#     print(f"\nTotal valid phase space points: {len(coordinates_list)}")
    
#     # Compare with batch reconstruction
#     print("\nBatch reconstruction...")
#     psr_batch = RealtimeTimeDelayMethodPSR(embedding_dimension=3, time_delay=10)
#     phase_space = psr_batch.reconstruct(signal)
#     print(f"Phase space shape: {phase_space.shape}")
#     print(f"First few points:\n{phase_space[:3]}")