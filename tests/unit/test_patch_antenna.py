"""Unit tests for PatchAntenna geometry."""

import pytest
from emsim.geometry.patch_antenna import PatchAntenna


def test_patch_antenna_creation():
    """Test basic PatchAntenna instantiation."""
    patch = PatchAntenna(
        patch_width=32e-3,
        patch_length=40e-3,
        substrate_width=60e-3,
        substrate_length=60e-3,
        substrate_thickness=1.524e-3,
        substrate_eps_r=3.38,
        substrate_kappa=1e-3,
        feed_x=-6e-3,
        sim_box=(200e-3, 200e-3, 150e-3)
    )
    
    assert patch.patch_width == 32e-3
    assert patch.patch_length == 40e-3
    assert patch.substrate_eps_r == 3.38
    assert patch.feed_x == -6e-3


def test_patch_antenna_ranges():
    """Test spatial range properties."""
    patch = PatchAntenna(
        patch_width=32e-3,
        patch_length=40e-3,
        substrate_width=60e-3,
        substrate_length=60e-3,
        substrate_thickness=1.524e-3,
        substrate_eps_r=3.38,
        substrate_kappa=1e-3,
        feed_x=-6e-3,
        sim_box=(200e-3, 200e-3, 150e-3)
    )
    
    x_range = patch.x_range
    y_range = patch.y_range
    z_range = patch.z_range
    
    assert len(x_range) == 2
    assert len(y_range) == 2
    assert len(z_range) == 2
    
    # Check symmetry
    assert x_range[0] == -100e-3
    assert x_range[1] == 100e-3
    assert y_range[0] == -100e-3
    assert y_range[1] == 100e-3
    
    # Check z range (ground at z=0)
    assert z_range[0] < 0  # Air below
    assert z_range[1] > 0  # Air above


def test_patch_antenna_dataclass():
    """Test that PatchAntenna is a proper dataclass."""
    from dataclasses import is_dataclass
    
    assert is_dataclass(PatchAntenna)
