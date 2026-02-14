"""Tests for physical constants."""

import math
import pytest
from emsim.constants import EPS0, MU0, C0, ETA0


def test_speed_of_light():
    expected = 1.0 / math.sqrt(EPS0 * MU0)
    assert abs(C0 - expected) / expected < 1e-10


def test_speed_of_light_value():
    assert abs(C0 - 299_792_458.0) / 299_792_458.0 < 1e-6


def test_impedance():
    expected = math.sqrt(MU0 / EPS0)
    assert abs(ETA0 - expected) / expected < 1e-10


def test_impedance_value():
    assert abs(ETA0 - 376.73) / 376.73 < 1e-3
