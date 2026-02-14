"""Base material types for the EMSIM FDTD electromagnetic simulator.

This module defines isotropic (Material), dispersive (DispersiveMaterial),
and anisotropic (AnisotropicMaterial) electromagnetic material models.
All docstrings and metadata are in English for consistency with the project
documentation standards.
"""

from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any


@dataclass(frozen=True)
class Material:
    """Isotropic, non-dispersive electromagnetic material.

    Represents a linear material with constant relative permittivity (eps_r),
    relative permeability (mu_r), and conductivity (sigma). Suitable for
    vacuum, dielectrics, and DC conductor approximations.

    Parameters
    ----------
    name : str
        Human-readable material name (e.g. "FR-4", "Copper").
    eps_r : float
        Relative permittivity (dimensionless). Must be >= 1.0 for passive media.
    mu_r : float, optional
        Relative permeability (dimensionless). Default 1.0 (non-magnetic).
    sigma : float, optional
        Electrical conductivity [S/m]. Default 0.0 (lossless).
    category : str, optional
        Material category for filtering. One of: "dielectric", "conductor",
        "magnetic", "biological", "semiconductor", "reference", "custom".
        Default "custom".
    description : str, optional
        Short description or application note (e.g. frequency range, source).
    source : str, optional
        Reference (datasheet, paper, or standard) for the parameter values.

    Attributes
    ----------
    name, eps_r, mu_r, sigma, category, description, source
        As in Parameters.

    Examples
    --------
    >>> mat = Material("Vacuum", eps_r=1.0, mu_r=1.0, sigma=0.0, category="reference")
    >>> mat.eps_r
    1.0
    >>> mat.to_dict()
    {'name': 'Vacuum', 'eps_r': 1.0, ...}

    Notes
    -----
    FDTD update coefficients (Ca, Cb, dt_over_mu) are computed from these
    values in MaterialGrid.compute_coefficients(dt).
    """

    name: str
    eps_r: float
    mu_r: float = 1.0
    sigma: float = 0.0
    category: Literal[
        "dielectric", "conductor", "magnetic", "biological",
        "semiconductor", "reference", "custom"
    ] = "custom"
    description: str = ""
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Return material properties as a dictionary (e.g. for JSON/CSV export)."""
        return {k: v for k, v in self.__dict__.items()}

    def __str__(self) -> str:
        return (
            f"{self.name} (eps_r={self.eps_r}, mu_r={self.mu_r}, "
            f"sigma={self.sigma} S/m)"
        )


@dataclass(frozen=True)
class DispersiveMaterial(Material):
    """Frequency-dispersive electromagnetic material (Drude, Lorentz, or Debye).

    Extends Material with frequency-dependent permittivity. Used for metals
    (Drude), resonant dielectrics (Lorentz), or relaxation (Debye). The
    FDTD implementation uses auxiliary differential equations (ADE) with
    polarization P and/or current density J.

    Parameters
    ----------
    name : str
        Material name.
    eps_r : float
        Not used for dispersive models; kept for interface compatibility.
        Use eps_inf (Drude/Lorentz) or eps_s (Debye) instead.
    mu_r : float, optional
        Relative permeability. Default 1.0.
    sigma : float, optional
        DC conductivity added to model if needed. Default 0.0.
    dispersive : bool, optional
        Flag for dispersive model. Default True.
    model : str, optional
        One of "drude", "lorentz", "debye". Default "drude".

    Drude model (metals, plasmas)
    -----------------------------
    eps(omega) = eps_inf - omega_p^2 / (omega^2 + i*gamma*omega)
    eps_inf : float, optional
        High-frequency permittivity.
    omega_p : float, optional
        Plasma frequency [rad/s].
    gamma : float, optional
        Collision frequency [rad/s].

    Lorentz model (resonances)
    --------------------------
    delta_eps : list of float, optional
        Strength of each Lorentz pole.
    omega_0 : list of float, optional
        Resonant frequency [rad/s] per pole.
    delta : list of float, optional
        Damping (linewidth) [rad/s] per pole.

    Debye model (relaxation)
    ------------------------
    eps_s : float, optional
        Static permittivity.
    tau : float, optional
        Relaxation time [s].

    category, description, source
        As in Material.

    References
    ----------
    .. [1] Taflove & Hagness, "Computational Electrodynamics", 3rd ed.,
           Ch. 9 (Dispersive Media).
    """

    dispersive: bool = True
    model: Literal["drude", "lorentz", "debye"] = "drude"

    # Drude: eps(w) = eps_inf - wp^2/(w^2 + i*gamma*w)
    eps_inf: Optional[float] = None
    omega_p: Optional[float] = None
    gamma: Optional[float] = None

    # Lorentz: multiple poles
    delta_eps: Optional[tuple] = None
    omega_0: Optional[tuple] = None
    delta: Optional[tuple] = None

    # Debye
    eps_s: Optional[float] = None
    tau: Optional[float] = None


@dataclass(frozen=True)
class AnisotropicMaterial(Material):
    """Anisotropic material with 3x3 permittivity and permeability tensors.

    Used for uniaxial crystals, liquid crystals, and magnetized plasmas.
    Relative permittivity and permeability are symmetric 3x3 tensors in the
    grid coordinate system (x, y, z).

    Parameters
    ----------
    name : str
        Material name.
    eps_r : float
        Ignored; use eps_r_xx, eps_r_yy, eps_r_zz and off-diagonal terms.
    mu_r : float
        Ignored; use mu_r_xx, mu_r_yy, mu_r_zz and off-diagonal terms.
    sigma : float, optional
        Isotropic conductivity. Default 0.0.
    anisotropic : bool, optional
        Flag. Default True.

    Permittivity tensor (relative)
    ------------------------------
    eps_r_xx, eps_r_yy, eps_r_zz : float, optional
        Diagonal components. Default 1.0.
    eps_r_xy, eps_r_xz, eps_r_yz : float, optional
        Off-diagonal (symmetric). Default 0.0.

    Permeability tensor (relative)
    ------------------------------
    mu_r_xx, mu_r_yy, mu_r_zz : float, optional
        Diagonal. Default 1.0.
    mu_r_xy, mu_r_xz, mu_r_yz : float, optional
        Off-diagonal. Default 0.0.

    category, description, source
        As in Material.

    Notes
    -----
    The FDTD update for anisotropic media requires solving a linear system
    E_new = eps^{-1} * (D_updated) at each cell where the material is
    anisotropic.
    """

    anisotropic: bool = True

    eps_r_xx: float = 1.0
    eps_r_yy: float = 1.0
    eps_r_zz: float = 1.0
    eps_r_xy: float = 0.0
    eps_r_xz: float = 0.0
    eps_r_yz: float = 0.0

    mu_r_xx: float = 1.0
    mu_r_yy: float = 1.0
    mu_r_zz: float = 1.0
    mu_r_xy: float = 0.0
    mu_r_xz: float = 0.0
    mu_r_yz: float = 0.0
