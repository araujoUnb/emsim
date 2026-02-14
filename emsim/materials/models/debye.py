"""Debye model for dielectric relaxation in FDTD.

Permittivity: eps(omega) = eps_inf + (eps_s - eps_inf) / (1 + i*omega*tau).
Single-pole relaxation. FDTD uses polarization P with first-order ADE.
"""

from ..base import DispersiveMaterial


def DebyeMaterial(
    name: str,
    eps_s: float,
    eps_inf: float,
    tau: float,
    mu_r: float = 1.0,
    sigma: float = 0.0,
    category: str = "dielectric",
    description: str = "",
    source: str = "",
) -> DispersiveMaterial:
    """Create a Debye dispersive material (relaxation).

    Parameters
    ----------
    name : str
        Material name.
    eps_s : float
        Static (low-frequency) relative permittivity.
    eps_inf : float
        High-frequency relative permittivity.
    tau : float
        Relaxation time [s].
    mu_r : float, optional
        Relative permeability. Default 1.0.
    sigma : float, optional
        Conductivity [S/m]. Default 0.0.
    category : str, optional
        Category. Default "dielectric".
    description : str, optional
        Short description.
    source : str, optional
        Reference.

    Returns
    -------
    DispersiveMaterial
        Instance with model="debye".
    """
    return DispersiveMaterial(
        name=name,
        eps_r=eps_s,
        mu_r=mu_r,
        sigma=sigma,
        dispersive=True,
        model="debye",
        eps_inf=eps_inf,
        eps_s=eps_s,
        tau=tau,
        category=category,
        description=description,
        source=source,
    )
