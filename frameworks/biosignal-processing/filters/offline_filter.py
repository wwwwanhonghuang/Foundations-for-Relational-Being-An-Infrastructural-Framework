
import numpy as np
from scipy import signal as sp_signal
from typing import Optional, Tuple
from filters.base_filter import BaseFilter


class OfflineFilter(BaseFilter):
    """Base class for offline/batch filtering."""
    
    def __init__(self):
        super().__init__()
    
    def filter(self, signal):
        """Filter entire signal at once."""
        raise NotImplementedError("Subclasses must implement filter method")
