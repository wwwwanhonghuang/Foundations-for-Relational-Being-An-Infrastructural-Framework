
import numpy as np
from scipy import signal as sp_signal
from typing import Optional, Tuple
from filters.base_filter import BaseFilter

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

