import os
import numpy as np
from openEMS import openEMS
from openEMS.physical_constants import C0
from CSXCAD import ContinuousStructure
from .fdtd_base_class import FDTDBaseClass

class FDTDAntenna(FDTDBaseClass):
    def __init__(self, antenna, sim_path="/tmp/FDTDAntenna", f0=2e9, fc=1e9):
        """
        Initialize the FDTDAntenna class with an antenna object and simulation settings.
        
        Parameters:
        - antenna: the antenna object (e.g., a patch antenna) with geometry and CSX structure.
        - sim_path: the directory where the simulation files will be saved.
        - f0: center frequency of the excitation signal (in Hz).
        - fc: bandwidth of the excitation signal (in Hz).
        """
        super().__init__(antenna, sim_path, f0, fc)  # Initialize the base class

        self.CSX = self.antenna.CSX  # Link to the antenna's CSX structure
        self.FDTD = None  # Will hold the openEMS simulation object
        self.mesh = self.CSX.GetGrid()  # Get the simulation grid
        self.mesh.SetDeltaUnit(1e-3)  # Set mesh unit to millimeters
        self.mesh_res = C0 / (f0 + fc) / 1e-3 / 20  # Calculate mesh resolution in mm
        self.port = None  # Will store the feed port reference
        self.nf2ff_box = None  # Will store the NF2FF box for far-field analysis

    def configure_simulation(self):
        """
        Configure the FDTD simulation settings.
        - Sets excitation waveform.
        - Sets boundary conditions.
        - Links the CSX structure.
        """
        self.FDTD = openEMS(NrTS=30000, EndCriteria=1e-4)  # Max 30,000 time steps, -40 dB end criteria
        self.FDTD.SetGaussExcite(self.f0, self.fc)  # Gaussian excitation centered at f0 with bandwidth fc
        self.FDTD.SetBoundaryCond(['MUR'] * 6)  # Use MUR absorbing boundaries on all 6 sides
        self.FDTD.SetCSX(self.CSX)  # Link the geometry (CSX) with the simulator

    def setup_simulation_domain(self, margin=140):
        """
        Define the simulation domain (air box) and refine the mesh in the z-direction.
        
        Parameters:
        - margin: extra space added around the antenna (in mm).
        """
        # Total simulation domain size
        size_x = self.antenna.substrate_width + margin
        size_y = self.antenna.substrate_length + margin
        size_z = 150  # Fixed height of the simulation domain

        # Add coarse initial mesh lines in each direction
        self.mesh.AddLine('x', [-size_x / 2, size_x / 2])
        self.mesh.AddLine('y', [-size_y / 2, size_y / 2])
        self.mesh.AddLine('z', [-size_z / 3, size_z * 2 / 3])

        # Refine mesh lines within the substrate thickness
        substrate_cells = 4  # Number of divisions across the thickness
        self.mesh.AddLine('z', np.linspace(0, self.antenna.substrate_thickness, substrate_cells + 1))
        
        # 🔧 Smooth mesh for better accuracy and numerical stability
        self.mesh.SmoothMeshLines('all', self.mesh_res, 1.4)

    def add_geometry_to_grid(self):
        """
        Add the antenna geometry to the simulation mesh grid.
        This includes metallic edges for patch and ground.
        """
        self.FDTD.AddEdges2Grid(dirs='xy', properties=self.antenna.patch, metal_edge_res=self.mesh_res / 2)
        self.FDTD.AddEdges2Grid(dirs='xy', properties=self.antenna.gnd)

    def add_feed_port(self, feed_R=50):
        """
        Add a lumped port to excite the antenna.
        
        Parameters:
        - feed_R: feed resistance in Ohms (usually 50 Ω).
        """
        feed = self.antenna.geometry['feed']  # Get feed location from antenna geometry
        start = feed['start']
        stop = feed['stop']

        # Add a lumped port in the z-direction between 'start' and 'stop'
        self.port = self.FDTD.AddLumpedPort(
            1,         # Port number
            feed_R,    # Resistance
            start,     # Start position of the feed
            stop,      # Stop position of the feed
            'z',       # Excitation direction
            1.0,       # Voltage amplitude
            priority=5,
            edges2grid='xy'
        )

    def add_nf2ff_box(self):
        """
        Create and attach a near-field to far-field transformation box.
        This allows calculation of the far-field radiation pattern.
        """
        self.nf2ff_box = self.FDTD.CreateNF2FFBox()
        print("📦 NF2FF box added to simulation.")

    def run_simulation(self, cleanup=True):
        """
        Execute the FDTD simulation and optionally clean up temporary files.

        Parameters:
        - cleanup: if True, removes intermediate files after simulation.
        """
        # Ensure output directory exists
        if not os.path.exists(self.sim_path):
            os.makedirs(self.sim_path)

        print("▶️ Running openEMS simulation...")
        self.FDTD.Run(self.sim_path, cleanup=cleanup)  # Start simulation
        print("✅ Simulation completed.")

    def export_to_xml(self, filename="fdtd_sim.xml"):
        """
        Export the current simulation geometry to an XML file.

        Parameters:
        - filename: the name of the XML file to save.
        """
        filepath = os.path.join(self.sim_path, filename)
        os.makedirs(self.sim_path, exist_ok=True)  # Ensure folder exists
        self.CSX.Write2XML(filepath)  # Export geometry to XML
        print(f"📝 Geometry exported to: {filepath}")

    def get_nf2ff_box(self):
        """
        Returns the NF2FF box instance for post-processing (e.g., far-field calculations).
        """
        return self.nf2ff_box
