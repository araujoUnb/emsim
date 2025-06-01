# simulator.py

import abc

class Simulator(abc.ABC):
    """
    Abstract base class for any electromagnetic simulator.
    Subclasses must implement `run()` to carry out a simulation,
    and may return field data or post-processed quantities (e.g. S-parameters).
    """

    @abc.abstractmethod
    def __init__(self, geometry, **kwargs):
        """
        geometry: an instance of a 3D object (e.g. Object3D, RectangularWaveguide, etc.)
        Any additional keyword arguments can be used by subclasses (e.g. time‐step factor, boundary type).
        """
        self.geometry = geometry

    @abc.abstractmethod
    def run(self, N_steps: int, record_interval: int = 1):
        """
        Run the simulation for `N_steps` time steps.
        `record_interval` controls how often (every N steps) data is recorded.

        Returns:
            A tuple containing any data the simulator deems useful. For example:
            - Ez_record: a list (or tensor) of Ez-plane snapshots every record_interval steps
            - freq_axis: frequency axis if computing FFT
            - S11, S21: reflection/transmission coefficients (optional)
        """
        raise NotImplementedError("Subclasses must implement run().")
