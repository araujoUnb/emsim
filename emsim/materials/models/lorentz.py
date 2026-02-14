"""Lorentz model for resonant dielectrics in FDTD.

Permittivity: eps(omega) = eps_inf + sum_p delta_eps_p * omega_0_p^2 / (omega_0_p^2 - omega^2 - i*delta_p*omega).
Single-pole or multi-pole. FDTD uses polarization P with second-order ADE.
"""

from typing import Optional, Sequence, Union

from ..base import DispersiveMaterial


def LorentzMaterial(
    name: str,
    eps_inf: float,
    delta_eps: Union[float, Sequence[float]],
    omega_0: Union[float, Sequence[float]],
    delta: Union[float, Sequence[float]],
    mu_r: float = 1.0,
    sigma: float = 0.0,
    category: str = "dielectric",
    description: str = "",
    source: str = "",
) -> DispersiveMaterial:
    """Create a Lorentz dispersive material (resonant dielectric).

    Parameters
    ----------
    name : str
        Material name.
    eps_inf : float
        High-frequency relative permittivity.
    delta_eps : float or sequence of float
        Strength of each Lorentz pole (one or more).
    omega_0 : float or sequence of float
        Resonant frequency [rad/s] per pole.
    delta : float or sequence of float
        Damping (linewidth) [rad/s] per pole.
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
        Instance with model="lorentz".
    """
    def _tuple(v: Union[float, Sequence[float]]) -> tuple:
        return (v,) if isinstance(v, (int, float)) else tuple(v)

    return DispersiveMaterial(
        name=name,
        eps_r=eps_inf,
        mu_r=mu_r,
        sigma=sigma,
        dispersive=True,
        model="lorentz",
        eps_inf=eps_inf,
        delta_eps=_tuple(delta_eps),
        omega_0=_tuple(omega_0),
        delta=_tuple(delta),
        category=category,
        description=description,
        source=source,
    )
