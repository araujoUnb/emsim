"""Patch antenna geometry definition."""

from dataclasses import dataclass


@dataclass
class PatchAntenna:
    """A microstrip patch antenna on a dielectric substrate.
    
    Consists of:
    - Rectangular metallic patch (PEC)
    - Dielectric substrate layer
    - Ground plane (PEC)
    - Air box for radiation
    
    Parameters
    ----------
    patch_width : float
        Width of the patch in x-direction [m].
    patch_length : float
        Length of the patch in y-direction [m].
    substrate_width : float
        Width of the substrate in x-direction [m].
    substrate_length : float
        Length of the substrate in y-direction [m].
    substrate_thickness : float
        Thickness of the substrate in z-direction [m].
    substrate_eps_r : float
        Relative permittivity of the substrate.
    substrate_kappa : float
        Conductivity (loss tangent) of the substrate [S/m].
    feed_x : float
        X-position of the feed point relative to patch center [m].
        Negative values are towards -x edge.
    sim_box : tuple[float, float, float]
        Total simulation domain size (x, y, z) [m].
    
    Examples
    --------
    2.4 GHz patch antenna on FR-4:
    >>> patch = PatchAntenna(
    ...     patch_width=32e-3,       # 32 mm
    ...     patch_length=40e-3,      # 40 mm
    ...     substrate_width=60e-3,   # 60 mm
    ...     substrate_length=60e-3,  # 60 mm
    ...     substrate_thickness=1.524e-3,  # 1.524 mm
    ...     substrate_eps_r=3.38,
    ...     substrate_kappa=1e-3,
    ...     feed_x=-6e-3,            # 6 mm from center towards -x
    ...     sim_box=(200e-3, 200e-3, 150e-3)
    ... )
    """
    patch_width: float
    patch_length: float
    substrate_width: float
    substrate_length: float
    substrate_thickness: float
    substrate_eps_r: float
    substrate_kappa: float
    feed_x: float
    sim_box: tuple[float, float, float]
    
    @property
    def x_range(self) -> tuple[float, float]:
        """Simulation domain range in x [m]."""
        return (-self.sim_box[0] / 2, self.sim_box[0] / 2)
    
    @property
    def y_range(self) -> tuple[float, float]:
        """Simulation domain range in y [m]."""
        return (-self.sim_box[1] / 2, self.sim_box[1] / 2)
    
    @property
    def z_range(self) -> tuple[float, float]:
        """Simulation domain range in z [m].
        
        Convention: ground plane at z=0, substrate extends to z=thickness,
        patch at z=thickness. Air above and below (with PML).
        """
        # Place ground at z=0, extend air below and above
        return (-self.sim_box[2] / 3, 2 * self.sim_box[2] / 3)
