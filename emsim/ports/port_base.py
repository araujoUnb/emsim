"""Base protocol for all port types in FDTD simulations.

This module defines the common interface that all port implementations must follow,
enabling the FDTDSolver to work with different port types (modal, lumped, etc.)
without modification.
"""

from typing import Protocol, Dict, Any, runtime_checkable


@runtime_checkable
class PortBase(Protocol):
    """Protocol defining the common interface for all port types.
    
    All port implementations (Port, LumpedPort, etc.) should implement these
    methods to be compatible with the FDTDSolver.
    
    Attributes
    ----------
    name : str
        Unique identifier for this port.
    """
    
    name: str
    
    def record(self, *args, **kwargs) -> None:
        """Record field values at the current time step.
        
        The specific arguments depend on the port type:
        - Modal ports: record(Ey, Hx, dy, dx)
        - Lumped ports: record(E_field, H_tangential_1, H_tangential_2, dl, ds)
        """
        ...
    
    def reset(self) -> None:
        """Clear all temporal records.
        
        Called before starting a new simulation run.
        """
        ...
    
    def compute_result(self, dt: float, **kwargs) -> Dict[str, Any]:
        """Compute final results from recorded temporal data.
        
        Parameters
        ----------
        dt : float
            Time step [s] between consecutive recordings.
        **kwargs
            Additional parameters specific to the port type (e.g., Z_mode_func
            for modal ports, frequency range for lumped ports).
        
        Returns
        -------
        dict
            Results dictionary with at least:
            - 'type': str indicating port type ('modal', 'lumped', etc.)
            Additional keys depend on the port type.
        """
        ...
