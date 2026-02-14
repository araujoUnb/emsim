"""Built-in material catalog for the EMSIM FDTD simulator.

Provides a dictionary of common electromagnetic materials (dielectrics,
conductors, biological tissues) for use with MaterialManager. All entries
include name, eps_r, mu_r, sigma, category, and optional description/source.
Dispersive materials (e.g. copper_drude) use the Drude model for RF/optical
applications.
"""

from .base import Material, DispersiveMaterial


MATERIAL_CATALOG = {
    # Reference
    "vacuum": Material(
        "Vacuum",
        eps_r=1.0,
        mu_r=1.0,
        sigma=0.0,
        category="reference",
        description="Perfect vacuum; reference for permittivity and permeability.",
    ),
    "air": Material(
        "Air",
        eps_r=1.00059,
        mu_r=1.0,
        sigma=0.0,
        category="dielectric",
        description="Dry air at standard conditions.",
    ),

    # RF substrates
    "fr4": Material(
        "FR-4",
        eps_r=4.4,
        mu_r=1.0,
        sigma=0.02,
        category="dielectric",
        description="PCB substrate (glass-reinforced epoxy); loss tangent ~0.02.",
    ),
    "rogers_ro4003c": Material(
        "Rogers RO4003C",
        eps_r=3.38,
        mu_r=1.0,
        sigma=0.0,
        category="dielectric",
        description="High-frequency laminate for RF applications.",
        source="Rogers Corp datasheet",
    ),
    "rt5880": Material(
        "Rogers RT/duroid 5880",
        eps_r=2.2,
        mu_r=1.0,
        sigma=0.0,
        category="dielectric",
        description="Low-loss PTFE composite.",
    ),
    "alumina": Material(
        "Alumina 99.5%",
        eps_r=9.8,
        mu_r=1.0,
        sigma=1e-12,
        category="dielectric",
        description="High-purity ceramic substrate.",
    ),
    "ptfe": Material(
        "PTFE Teflon",
        eps_r=2.1,
        mu_r=1.0,
        sigma=0.0,
        category="dielectric",
        description="Polytetrafluoroethylene.",
    ),
    "silicon": Material(
        "Silicon",
        eps_r=11.7,
        mu_r=1.0,
        sigma=0.0,
        category="semiconductor",
        description="Pure silicon (high resistivity).",
    ),
    "gaas": Material(
        "GaAs",
        eps_r=12.9,
        mu_r=1.0,
        sigma=0.0,
        category="semiconductor",
        description="Gallium arsenide.",
    ),

    # Conductors (DC approximation and Drude)
    "copper_dc": Material(
        "Copper (DC)",
        eps_r=1.0,
        mu_r=0.999991,
        sigma=5.96e7,
        category="conductor",
        description="DC conductivity approximation; use at low frequency.",
    ),
    "copper_drude": DispersiveMaterial(
        name="Copper (Drude)",
        eps_r=1.0,
        mu_r=1.0,
        sigma=0.0,
        model="drude",
        eps_inf=1.0,
        omega_p=1.63e16,
        gamma=4.1e13,
        category="conductor",
        description="Drude model for RF and optical frequencies.",
    ),
    "gold": Material(
        "Gold",
        eps_r=1.0,
        mu_r=1.0,
        sigma=4.1e7,
        category="conductor",
        description="DC approximation.",
    ),
    "aluminum": Material(
        "Aluminum",
        eps_r=1.0,
        mu_r=1.000022,
        sigma=3.77e7,
        category="conductor",
    ),
    "silver": Material(
        "Silver",
        eps_r=1.0,
        mu_r=1.0,
        sigma=6.3e7,
        category="conductor",
    ),

    # Water and biological (typical values at 2.45 GHz)
    "water": Material(
        "Water (distilled)",
        eps_r=80.0,
        mu_r=1.0,
        sigma=5e-6,
        category="dielectric",
    ),
    "muscle": Material(
        "Muscle tissue",
        eps_r=54.8,
        mu_r=1.0,
        sigma=1.74,
        category="biological",
        description="Typical values at 2.45 GHz.",
    ),
    "skin_dry": Material(
        "Skin (dry)",
        eps_r=38.0,
        mu_r=1.0,
        sigma=1.46,
        category="biological",
        description="Typical values at 2.45 GHz.",
    ),

    # Test / custom
    "lossy_test": Material(
        "Lossy test",
        eps_r=2.5,
        mu_r=1.0,
        sigma=0.1,
        category="custom",
        description="For validation and unit tests.",
    ),
}
