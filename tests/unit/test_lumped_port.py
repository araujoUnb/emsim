"""Unit tests for LumpedPort class."""

import pytest
import numpy as np
import tensorflow as tf

from emsim.ports.lumped_port import LumpedPort


def test_lumped_port_creation():
    """Test basic LumpedPort instantiation."""
    port = LumpedPort(
        name="test_port",
        i=10, j=20, k=5,
        direction='z',
        resistance=50.0
    )
    
    assert port.name == "test_port"
    assert port.position == (10, 20, 5)
    assert port.direction == 'z'
    assert port.resistance == 50.0
    assert len(port.V_record) == 0
    assert len(port.I_record) == 0


def test_lumped_port_invalid_direction():
    """Test that invalid direction raises ValueError."""
    with pytest.raises(ValueError, match="direction must be"):
        LumpedPort("bad", i=0, j=0, k=0, direction='w', resistance=50.0)


def test_lumped_port_inject():
    """Test voltage injection into field."""
    port = LumpedPort("test", i=5, j=5, k=5, direction='z', resistance=50.0)
    
    # Create a test field
    Ez = tf.Variable(tf.zeros([10, 10, 10], dtype=tf.float32))
    
    # Inject voltage
    amplitude = 1.0
    dl = 1e-3
    port.inject(Ez, amplitude, dl)
    
    # Check that field was modified at port location
    injected_value = Ez[5, 5, 5].numpy()
    expected_value = amplitude / dl
    assert np.isclose(injected_value, expected_value)


def test_lumped_port_record():
    """Test recording of voltage and current."""
    port = LumpedPort("test", i=5, j=5, k=5, direction='z', resistance=50.0)
    
    # Create test fields
    Ez = tf.Variable(tf.ones([10, 10, 10], dtype=tf.float32) * 2.0)
    Hx = tf.Variable(tf.ones([10, 10, 10], dtype=tf.float32) * 0.1)
    Hy = tf.Variable(tf.ones([10, 10, 10], dtype=tf.float32) * 0.2)
    
    dl = 1e-3
    ds = 1e-6
    
    port.record(Ez, Hx, Hy, dl, ds)
    
    assert len(port.V_record) == 1
    assert len(port.I_record) == 1
    assert np.isclose(port.V_record[0], 2.0 * dl)


def test_lumped_port_reset():
    """Test resetting of records."""
    port = LumpedPort("test", i=0, j=0, k=0, direction='z', resistance=50.0)
    
    # Add some dummy data
    port.V_record = [1.0, 2.0, 3.0]
    port.I_record = [0.1, 0.2, 0.3]
    
    port.reset()
    
    assert len(port.V_record) == 0
    assert len(port.I_record) == 0


def test_lumped_port_compute_result():
    """Test impedance computation from recorded data."""
    port = LumpedPort("test", i=0, j=0, k=0, direction='z', resistance=50.0)
    
    # Create sinusoidal V and I signals
    dt = 1e-12
    n_steps = 100
    freq_sig = 2.4e9
    
    for n in range(n_steps):
        t = n * dt
        V = np.cos(2 * np.pi * freq_sig * t)
        I = 0.02 * np.cos(2 * np.pi * freq_sig * t)  # Z = V/I = 50 Ohm
        port.V_record.append(V)
        port.I_record.append(I)
    
    result = port.compute_result(dt=dt)
    
    assert result['type'] == 'lumped'
    assert 'freqs' in result
    assert 'Z_in' in result
    assert 'S11' in result
    assert len(result['freqs']) > 0
    assert len(result['Z_in']) == len(result['freqs'])


def test_lumped_port_protocol_compliance():
    """Test that LumpedPort implements PortBase protocol."""
    from emsim.ports.port_base import PortBase
    
    port = LumpedPort("test", i=0, j=0, k=0, direction='z', resistance=50.0)
    
    # Check protocol compliance
    assert isinstance(port, PortBase)
    assert hasattr(port, 'name')
    assert hasattr(port, 'record')
    assert hasattr(port, 'reset')
    assert hasattr(port, 'compute_result')
