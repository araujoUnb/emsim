"""Unit tests for modal Port class."""

import pytest
import numpy as np
import tensorflow as tf

from emsim.ports.port import Port


def test_port_creation():
    """Test basic Port instantiation."""
    mode_E = tf.ones([10, 10], dtype=tf.float32)
    mode_H = tf.ones([10, 10], dtype=tf.float32) * 0.5
    
    port = Port(
        name="test_port",
        k_plane=5,
        mode_profile_E=mode_E,
        mode_profile_H=mode_H,
        direction=1
    )
    
    assert port.name == "test_port"
    assert port.k_plane == 5
    assert port.direction == 1
    assert len(port.E_record) == 0
    assert len(port.H_record) == 0


def test_port_record():
    """Test field recording with modal overlap."""
    mode_E = tf.ones([10, 10], dtype=tf.float32)
    mode_H = tf.ones([10, 10], dtype=tf.float32) * 0.5
    
    port = Port("test", k_plane=5, mode_profile_E=mode_E, 
                mode_profile_H=mode_H, direction=1)
    
    # Create test fields
    Ey = tf.Variable(tf.ones([20, 10, 10], dtype=tf.float32) * 2.0)
    Hx = tf.Variable(tf.ones([20, 10, 10], dtype=tf.float32) * 0.1)
    
    dy = 1e-3
    dx = 1e-3
    
    port.record(Ey, Hx, dy, dx)
    
    assert len(port.E_record) == 1
    assert len(port.H_record) == 1
    
    # Check overlap calculation: sum(2.0 * 1.0) * 100 * 1e-6
    expected_E = 2.0 * 100 * 1e-6
    assert np.isclose(port.E_record[0], expected_E, rtol=1e-4)


def test_port_reset():
    """Test resetting of records."""
    mode_E = tf.ones([5, 5], dtype=tf.float32)
    mode_H = tf.ones([5, 5], dtype=tf.float32)
    
    port = Port("test", k_plane=0, mode_profile_E=mode_E, 
                mode_profile_H=mode_H, direction=1)
    
    # Add dummy data
    port.E_record = [1.0, 2.0, 3.0]
    port.H_record = [0.1, 0.2, 0.3]
    
    port.reset()
    
    assert len(port.E_record) == 0
    assert len(port.H_record) == 0


def test_port_compute_result():
    """Test compute_result returns correct structure."""
    mode_E = tf.ones([5, 5], dtype=tf.float32)
    mode_H = tf.ones([5, 5], dtype=tf.float32)
    
    port = Port("test", k_plane=0, mode_profile_E=mode_E,
                mode_profile_H=mode_H, direction=1)
    
    # Add some dummy recordings
    port.E_record = [1.0, 2.0, 3.0]
    port.H_record = [0.1, 0.2, 0.3]
    
    result = port.compute_result(dt=1e-12)
    
    assert result['type'] == 'modal'
    assert 'E_record' in result
    assert 'H_record' in result
    assert result['E_record'] == port.E_record
    assert result['H_record'] == port.H_record


def test_port_protocol_compliance():
    """Test that Port implements PortBase protocol."""
    from emsim.ports.port_base import PortBase
    
    mode_E = tf.ones([5, 5], dtype=tf.float32)
    mode_H = tf.ones([5, 5], dtype=tf.float32)
    port = Port("test", k_plane=0, mode_profile_E=mode_E,
                mode_profile_H=mode_H, direction=1)
    
    # Check protocol compliance
    assert isinstance(port, PortBase)
    assert hasattr(port, 'name')
    assert hasattr(port, 'record')
    assert hasattr(port, 'reset')
    assert hasattr(port, 'compute_result')
