"""Tests for PEC boundary conditions."""

import pytest
import tensorflow as tf
from emsim.boundaries.pec import apply_pec


def _make_ones(Nz, Ny, Nx):
    return tf.Variable(tf.ones([Nz, Ny, Nx], dtype=tf.float32))


def test_pec_x_minus():
    Ex, Ey, Ez = _make_ones(5, 5, 5), _make_ones(5, 5, 5), _make_ones(5, 5, 5)
    apply_pec(Ex, Ey, Ez, {'x-'})
    assert float(tf.reduce_max(tf.abs(Ey[:, :, 0])).numpy()) == 0.0
    assert float(tf.reduce_max(tf.abs(Ez[:, :, 0])).numpy()) == 0.0
    # Ex at x- boundary should be untouched (normal component)
    assert float(Ex[0, 0, 0].numpy()) == 1.0


def test_pec_x_plus():
    Ex, Ey, Ez = _make_ones(5, 5, 5), _make_ones(5, 5, 5), _make_ones(5, 5, 5)
    apply_pec(Ex, Ey, Ez, {'x+'})
    assert float(tf.reduce_max(tf.abs(Ey[:, :, -1])).numpy()) == 0.0
    assert float(tf.reduce_max(tf.abs(Ez[:, :, -1])).numpy()) == 0.0


def test_pec_y_minus():
    Ex, Ey, Ez = _make_ones(5, 5, 5), _make_ones(5, 5, 5), _make_ones(5, 5, 5)
    apply_pec(Ex, Ey, Ez, {'y-'})
    assert float(tf.reduce_max(tf.abs(Ex[:, 0, :])).numpy()) == 0.0
    assert float(tf.reduce_max(tf.abs(Ez[:, 0, :])).numpy()) == 0.0


def test_pec_z_faces():
    Ex, Ey, Ez = _make_ones(5, 5, 5), _make_ones(5, 5, 5), _make_ones(5, 5, 5)
    apply_pec(Ex, Ey, Ez, {'z-', 'z+'})
    # z-: tangential = Ex, Ey
    assert float(tf.reduce_max(tf.abs(Ex[0, :, :])).numpy()) == 0.0
    assert float(tf.reduce_max(tf.abs(Ey[0, :, :])).numpy()) == 0.0
    # z+:
    assert float(tf.reduce_max(tf.abs(Ex[-1, :, :])).numpy()) == 0.0
    assert float(tf.reduce_max(tf.abs(Ey[-1, :, :])).numpy()) == 0.0


def test_pec_all_faces():
    Ex, Ey, Ez = _make_ones(5, 5, 5), _make_ones(5, 5, 5), _make_ones(5, 5, 5)
    apply_pec(Ex, Ey, Ez, {'x-', 'x+', 'y-', 'y+', 'z-', 'z+'})
    # Interior should still be 1
    assert float(Ex[2, 2, 2].numpy()) == 1.0
