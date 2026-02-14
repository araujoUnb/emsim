"""Drude model for metals and plasmas in FDTD.

Permittivity: eps(omega) = eps_inf - omega_p^2 / (omega^2 + i*gamma*omega).
FDTD implementation uses auxiliary differential equations (ADE) with
current density J and polarization P. See Taflove & Hagness, Ch. 9.
"""

from typing import Optional

from ..base import DispersiveMaterial


def DrudeMaterial(
    name: str,
    eps_inf: float = 1.0,
    omega_p: float = 1.0,
    gamma: float = 1.0,
    mu_r: float = 1.0,
    sigma: float = 0.0,
    category: str = "conductor",
    description: str = "",
    source: str = "",
) -> DispersiveMaterial:
    """Create a Drude dispersive material (metals, plasmas).

    Parameters
    ----------
    name : str
        Material name (e.g. "Copper Drude").
    eps_inf : float
        High-frequency relative permittivity (dimensionless).
    omega_p : float
        Plasma frequency [rad/s].
    gamma : float
        Collision (damping) frequency [rad/s].
    mu_r : float, optional
        Relative permeability. Default 1.0.
    sigma : float, optional
        Additional DC conductivity [S/m]. Default 0.0.
    category : str, optional
        Category for catalog filtering. Default "conductor".
    description : str, optional
        Short description or application note.
    source : str, optional
        Reference (datasheet, paper).

    Returns
    -------
    DispersiveMaterial
        Instance with model="drude" and the given parameters.

    Notes
    -----
    Conductivity at DC: sigma_DC = eps_0 * omega_p^2 / gamma.
    """
    return DispersiveMaterial(
        name=name,
        eps_r=eps_inf,
        mu_r=mu_r,
        sigma=sigma,
        dispersive=True,
        model="drude",
        eps_inf=eps_inf,
        omega_p=omega_p,
        gamma=gamma,
        category=category,
        description=description,
        source=source,
    )
