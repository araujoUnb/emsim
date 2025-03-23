import numpy as np
from CSXCAD import ContinuousStructure
import plotly.graph_objects as go
from openEMS.physical_constants import C0, EPS0

class RectangularPatch:
    def __init__(self,
                 fo=2.45e9,  # Operating frequency in Hz
                 patch_width=32,  # Patch width in mm
                 patch_length=40,  # Patch length in mm
                 substrate_thickness=1.524,  # Substrate thickness in mm
                 substrate_width=60,  # Substrate width in mm
                 substrate_length=60,  # Substrate length in mm
                 metal_thickness=0,  # Metal thickness in mm
                 feed_pos=-6.0):  # Feed position in mm
        self.fo = fo
        self.patch_width = patch_width
        self.patch_length = patch_length
        self.substrate_thickness = substrate_thickness
        self.substrate_width = substrate_width
        self.substrate_length = substrate_length
        self.metal_thickness = metal_thickness
        self.feed_pos = feed_pos

        # Continuous structure to store antenna components
        self.CSX = ContinuousStructure()
        self.geometry = {}  # Dictionary to store created geometries

        # Create antenna components
        self.create_dielectric()
        self.create_patch()
        self.create_ground()
        self.create_feed()

    def create_dielectric(self):
        """Create the dielectric substrate of the antenna."""
        epsR = 3.38  # Dielectric constant of the substrate
        kappa = 1e-3 * 2 * np.pi * self.fo * epsR * EPS0 # Substrate conductivity
        self.substrate = self.CSX.AddMaterial('substrate', epsilon=epsR, kappa=kappa)
        
        # Define start and stop coordinates for the substrate
        start = [-self.substrate_width / 2, -self.substrate_length / 2, 0]
        stop = [self.substrate_width / 2, self.substrate_length / 2, self.substrate_thickness]
        
        # Add the substrate as a box to the structure
        self.substrate.AddBox(priority=0, start=start, stop=stop)
        self.geometry['dielectric'] = {'start': start, 'stop': stop}

    def create_patch(self):
        """Create the metallic patch of the antenna."""
        self.patch = self.CSX.AddMetal('patch')
        
        # Define start and stop coordinates for the patch
        start = [-self.patch_width / 2, -self.patch_length / 2, self.substrate_thickness]
        stop = [self.patch_width / 2, self.patch_length / 2, self.substrate_thickness + self.metal_thickness]
        
        # Add the patch as a box to the structure
        self.patch.AddBox(priority=10, start=start, stop=stop)
        self.geometry['patch'] = {'start': start, 'stop': stop}

    def create_ground(self):
        """Create the ground plane of the antenna."""
        self.gnd = self.CSX.AddMetal('gnd')
        
        # Define start and stop coordinates for the ground plane
        start = [-self.substrate_width / 2, -self.substrate_length / 2, 0]
        stop = [self.substrate_width / 2, self.substrate_length / 2, self.metal_thickness]
        
        # Add the ground plane as a box to the structure
        self.gnd.AddBox(priority=10, start=start, stop=stop)
        self.geometry['ground'] = {'start': start, 'stop': stop}

    def create_feed(self):
        """Create the feed point of the antenna."""
        # Define the feed as a small cylinder
        feed_radius = 0.5  # Radius of the feed in mm
        start = [self.feed_pos, 0, 0]
        stop = [self.feed_pos, 0, self.substrate_thickness + self.metal_thickness]
        
        # Store the feed geometry
        self.geometry['feed'] = {
            'start': start,
            'stop': stop,
            'radius': feed_radius
        }

    def visualize(self):
        """
        Visualize the antenna geometry using Plotly with distinct colors for each component.
        """
        # Create a Plotly figure
        fig = go.Figure()

        # Define colors for each component
        colors = {
            'dielectric': 'lightgreen',  # Substrate
            'patch': 'blue',             # Patch
            'ground': 'gray',            # Ground plane
            'feed': 'gold'               # Feed
        }

        # Add each component to the figure
        for name, box in self.geometry.items():
            if name == 'feed':
                # Create a cylinder for the feed
                feed_radius = box['radius']
                start = box['start']
                stop = box['stop']
                z = np.linspace(start[2], stop[2], 50)
                theta = np.linspace(0, 2 * np.pi, 50)
                theta_grid, z_grid = np.meshgrid(theta, z)
                x_grid = start[0] + feed_radius * np.cos(theta_grid)
                y_grid = start[1] + feed_radius * np.sin(theta_grid)

                fig.add_trace(go.Surface(
                    x=x_grid,
                    y=y_grid,
                    z=z_grid,
                    colorscale=[[0, colors['feed']], [1, colors['feed']]],
                    showscale=False,
                    name=name
                ))
            else:
                # Create a box for other components
                start = box['start']
                stop = box['stop']
                dx = stop[0] - start[0]
                dy = stop[1] - start[1]
                dz = stop[2] - start[2]

                fig.add_trace(go.Mesh3d(
                    x=[start[0], stop[0], stop[0], start[0], start[0], stop[0], stop[0], start[0]],
                    y=[start[1], start[1], stop[1], stop[1], start[1], start[1], stop[1], stop[1]],
                    z=[start[2], start[2], start[2], start[2], stop[2], stop[2], stop[2], stop[2]],
                    i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                    j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                    k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                    color=colors[name],  # Assign color based on component
                    opacity=0.8,
                    name=name
                ))

        # Update layout for better visualization
        fig.update_layout(
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)'
            ),
            margin=dict(l=0, r=0, b=0, t=0)
        )

        # Show the figure
        fig.show()