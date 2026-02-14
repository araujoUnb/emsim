"""Unit tests for NF2FF functionality."""

import pytest
import numpy as np
import tensorflow as tf

from emsim.postprocessing.nf2ff import NF2FFBox, compute_nf2ff


def test_nf2ff_box_creation():
    """Test NF2FFBox instantiation."""
    box = NF2FFBox(
        i_range=(10, 50),
        j_range=(10, 50),
        k_range=(10, 50)
    )
    
    assert box.i_range == (10, 50)
    assert box.j_range == (10, 50)
    assert box.k_range == (10, 50)
    assert len(box.faces) == 6
    assert 'x-' in box.faces
    assert 'z+' in box.faces


def test_nf2ff_box_record():
    """Test recording fields on all faces."""
    box = NF2FFBox(
        i_range=(10, 20),
        j_range=(10, 20),
        k_range=(10, 20)
    )
    
    # Create test fields
    shape = [30, 30, 30]
    Ex = tf.Variable(tf.ones(shape, dtype=tf.float32))
    Ey = tf.Variable(tf.ones(shape, dtype=tf.float32) * 2)
    Ez = tf.Variable(tf.ones(shape, dtype=tf.float32) * 3)
    Hx = tf.Variable(tf.ones(shape, dtype=tf.float32) * 0.1)
    Hy = tf.Variable(tf.ones(shape, dtype=tf.float32) * 0.2)
    Hz = tf.Variable(tf.ones(shape, dtype=tf.float32) * 0.3)
    
    # Record
    box.record(Ex, Ey, Ez, Hx, Hy, Hz)
    
    # Check that data was recorded
    for face_name, face_data in box.faces.items():
        assert len(face_data['E_tan']) == 1
        assert len(face_data['H_tan']) == 1
        
        # Check that tangential components are present
        if face_name in ['x-', 'x+']:
            assert 'Ey' in face_data['E_tan'][0]
            assert 'Ez' in face_data['E_tan'][0]
        elif face_name in ['y-', 'y+']:
            assert 'Ex' in face_data['E_tan'][0]
            assert 'Ez' in face_data['E_tan'][0]
        elif face_name in ['z-', 'z+']:
            assert 'Ex' in face_data['E_tan'][0]
            assert 'Ey' in face_data['E_tan'][0]


def test_nf2ff_box_reset():
    """Test resetting recorded data."""
    box = NF2FFBox(
        i_range=(10, 20),
        j_range=(10, 20),
        k_range=(10, 20)
    )
    
    # Add dummy data
    box.faces['x-']['E_tan'].append({'Ey': np.ones((10, 10)), 'Ez': np.ones((10, 10))})
    box.faces['x-']['H_tan'].append({'Hy': np.ones((10, 10)), 'Hz': np.ones((10, 10))})
    
    box.reset()
    
    for face_data in box.faces.values():
        assert len(face_data['E_tan']) == 0
        assert len(face_data['H_tan']) == 0


def test_compute_nf2ff_structure():
    """Test that compute_nf2ff returns correct structure."""
    box = NF2FFBox(
        i_range=(10, 20),
        j_range=(10, 20),
        k_range=(10, 20)
    )
    
    # Add some dummy recordings
    for _ in range(10):
        for face_data in box.faces.values():
            face_data['E_tan'].append({'Ex': np.ones((10, 10)), 'Ey': np.ones((10, 10))})
            face_data['H_tan'].append({'Hx': np.ones((10, 10)), 'Hy': np.ones((10, 10))})
    
    # Compute nf2ff
    theta = np.arange(-180, 180, 10)
    phi = np.array([0, 90])
    grid_info = {'dx': 1e-3, 'dy': 1e-3, 'dz': 1e-3, 'dt': 1e-12}
    
    result = compute_nf2ff(box, freq=2.4e9, theta=theta, phi=phi, grid_info=grid_info)
    
    # Check result structure
    assert 'E_theta' in result
    assert 'E_phi' in result
    assert 'E_norm' in result
    assert 'directivity' in result
    assert 'Dmax' in result
    assert 'theta' in result
    assert 'phi' in result
    assert 'freq' in result
    
    # Check shapes
    assert result['E_theta'].shape == (len(theta), len(phi))
    assert result['E_phi'].shape == (len(theta), len(phi))
    assert result['directivity'].shape == (len(theta), len(phi))


def test_compute_nf2ff_placeholder_note():
    """Test that compute_nf2ff includes placeholder note."""
    box = NF2FFBox(i_range=(10, 20), j_range=(10, 20), k_range=(10, 20))
    
    result = compute_nf2ff(
        box, freq=2.4e9,
        theta=np.array([0, 90, 180]),
        phi=np.array([0, 90]),
        grid_info={'dx': 1e-3, 'dy': 1e-3, 'dz': 1e-3, 'dt': 1e-12}
    )
    
    assert 'note' in result
    assert 'placeholder' in result['note'].lower()
