from abc import ABC, abstractmethod

class FDTDBaseClass(ABC):
    def __init__(self, antenna, sim_path, f0, fc):
        """
        Abstract base class for FDTD simulations.

        Parameters
        ----------
        antenna : object
            object containing geometry and materials.
        sim_path : str
            Path to store simulation data.
        f0 : float
            Center frequency in Hz.
        fc : float
            Frequency bandwidth (20 dB) in Hz.
        """
        self.antenna = antenna
        self.sim_path = sim_path
        self.f0 = f0
        self.fc = fc

    @abstractmethod
    def configure_simulation(self):
        pass

    @abstractmethod
    def setup_simulation_domain(self):
        pass

    @abstractmethod
    def add_geometry_to_grid(self):
        pass

    @abstractmethod
    def add_feed_port(self):
        pass

    @abstractmethod
    def run_simulation(self, cleanup=True):
        pass

    @abstractmethod
    def export_to_xml(self, filename):
        pass
