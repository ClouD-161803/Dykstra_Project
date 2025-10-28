"""
This module defines a data class for storing projection solver results.

Classes:
- ProjectionResult:
    Container for projection solver outputs.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProjectionResult:
    """
    Data class for storing results from a projection solver.
    
    Attributes:
        projection: Final projected point.
        path: List of intermediate points visited during projection.
        squared_errors: Array of squared errors at each iteration.
        stalled_errors: Array of stalled error flags at each iteration.
        converged_errors: Array of converged error flags at each iteration.
        errors_for_plotting: List of error vectors for visualization.
        active_half_spaces: Matrix of active half-space flags.
    """
    projection: np.ndarray
    path: list
    squared_errors: Optional[np.ndarray] = None
    stalled_errors: Optional[np.ndarray] = None
    converged_errors: Optional[np.ndarray] = None
    errors_for_plotting: Optional[list] = None
    active_half_spaces: Optional[np.ndarray] = None
    
    def has_error_tracking(self) -> bool:
        """Check if error tracking data is available."""
        return self.squared_errors is not None
    
    def has_error_plotting_data(self) -> bool:
        """Check if error plotting data is available."""
        return self.errors_for_plotting is not None
    
    def has_active_halfspace_data(self) -> bool:
        """Check if active half-space data is available."""
        return self.active_half_spaces is not None
