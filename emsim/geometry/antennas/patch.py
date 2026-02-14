"""Patch antenna geometry definition."""

from dataclasses import dataclass
from typing import Any


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
        return (-self.sim_box[2] / 3, 2 * self.sim_box[2] / 3)

    def bounds(self) -> tuple[float, float, float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max, z_min, z_max) [m] for the simulation box."""
        x0, x1 = self.x_range
        y0, y1 = self.y_range
        z0, z1 = self.z_range
        return (x0, x1, y0, y1, z0, z1)

    def to_pyvista(self, **kwargs: Any):
        """Return PyVista meshes for substrate box and patch (as a MultiBlock or list)."""
        try:
            import pyvista as pv
        except ImportError as e:
            raise ImportError("PyVista is required for to_pyvista(). Install with: pip install pyvista") from e
        # Substrate: centered in x,y; z from 0 to substrate_thickness
        cx, cy = 0.0, 0.0
        sx, sy = self.substrate_width, self.substrate_length
        sub = pv.Box(
            bounds=(
                cx - sx / 2, cx + sx / 2,
                cy - sy / 2, cy + sy / 2,
                0.0, self.substrate_thickness,
            ),
            **kwargs,
        )
        sub.cell_data["type"] = ["substrate"] * sub.n_cells
        # Patch: centered; at z = substrate_thickness, size patch_width x patch_length
        px, py = self.patch_width, self.patch_length
        patch = pv.Box(
            bounds=(
                cx - px / 2, cx + px / 2,
                cy - py / 2, cy + py / 2,
                self.substrate_thickness, self.substrate_thickness + 1e-6,
            ),
            **kwargs,
        )
        patch.cell_data["type"] = ["patch"] * patch.n_cells
        return pv.MultiBlock([sub, patch])
