"""Backward compatibility: re-export PatchAntenna from antennas subpackage."""

from emsim.geometry.antennas import PatchAntenna

__all__ = ["PatchAntenna"]
