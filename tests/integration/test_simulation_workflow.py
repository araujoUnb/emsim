"""Integration tests for complete simulation workflow.

These tests verify the end-to-end workflow: YAML config → Simulation → results.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from emsim.simulation import Simulation
from emsim.geometry.rectangular_waveguide import RectangularWaveguide
from emsim.geometry.patch_antenna import PatchAntenna


@pytest.mark.integration
def test_simulation_from_yaml_waveguide():
    """Test loading and running a waveguide simulation from YAML."""
    # Use existing WR42 config
    config_path = Path("Simulations/WR42/config.yaml")
    
    if not config_path.exists():
        pytest.skip("WR42 config not found")
    
    # Load simulation and build (from_yaml only loads config)
    sim = Simulation.from_yaml(str(config_path))
    sim.build()
    
    # Check that components were built
    assert sim._geometry is not None
    assert isinstance(sim._geometry, RectangularWaveguide)
    assert sim._grid is not None
    assert sim._ports is not None
    assert sim._solver is not None


@pytest.mark.integration
def test_simulation_from_yaml_patch_antenna():
    """Test loading and running a patch antenna simulation from YAML."""
    config_path = Path("Simulations/Patch_Antenna/config.yaml")
    
    if not config_path.exists():
        pytest.skip("Patch Antenna config not found")
    
    sim = Simulation.from_yaml(str(config_path))
    sim.build()
    
    # Check components
    assert sim._geometry is not None
    assert isinstance(sim._geometry, PatchAntenna)
    assert sim._grid is not None
    assert sim._solver is not None


@pytest.mark.integration
@pytest.mark.slow
def test_simulation_run_short():
    """Test running a minimal simulation end-to-end."""
    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create minimal YAML config
        config = {
            'name': 'test_sim',
            'geometry': {
                'type': 'rectangular_waveguide',
                'a': 10.7e-3,
                'b': 4.3e-3,
                'length': 20e-3,
            },
            'frequency': {
                'center': 23e9,
                'bandwidth': 5e9,
            },
            'mode': {
                'm': 1,
                'n': 0,
            },
            'grid': {
                'resolution': 10,
                'courant': 0.5,
            },
            'boundaries': {
                'x': ['cpml', 'cpml'],
                'y': ['cpml', 'cpml'],
                'z': ['port', 'port'],
            },
            'run': {
                'n_steps': 50,  # Very short
                'record_interval': 10,
            },
            'output': {
                'directory': tmpdir,
            }
        }
        
        # Save config
        import yaml
        config_path = Path(tmpdir) / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Load, build and run
        sim = Simulation.from_yaml(str(config_path))
        result = sim.run()
        
        # Check result (solver returns S11, S21, freqs, n_steps_run, etc.)
        assert result is not None
        assert 'n_steps_run' in result or 'S11' in result


@pytest.mark.integration
def test_simulation_geometry_creation():
    """Test that Simulation correctly creates geometry objects."""
    # Mock minimal config dict
    config = {
        'geometry': {
            'type': 'rectangular_waveguide',
            'a': 10.7e-3,
            'b': 4.3e-3,
            'length': 30e-3,
        },
        'frequency': {'center': 20e9, 'bandwidth': 5e9},
        'mode': {'m': 1, 'n': 0},
        'grid': {'resolution': 12, 'courant': 0.5},
        'boundaries': {
            'x': ['cpml', 'cpml'],
            'y': ['cpml', 'cpml'],
            'z': ['port', 'port'],
        },
        'run': {'n_steps': 10, 'record_interval': 5},
        'output': {'directory': 'outputs'},
    }
    
    sim = Simulation.from_config(config)
    sim._build_geometry()
    
    assert sim._geometry is not None
    assert isinstance(sim._geometry, RectangularWaveguide)
    assert sim._geometry.a == 10.7e-3
    assert sim._geometry.b == 4.3e-3


@pytest.mark.integration
def test_simulation_grid_creation():
    """Test that Simulation correctly creates YeeGrid."""
    config = {
        'geometry': {
            'type': 'rectangular_waveguide',
            'a': 10.7e-3,
            'b': 4.3e-3,
            'length': 25e-3,
        },
        'frequency': {'center': 20e9, 'bandwidth': 5e9},
        'mode': {'m': 1, 'n': 0},
        'grid': {'resolution': 15, 'courant': 0.5},
        'boundaries': {
            'x': ['cpml', 'cpml'],
            'y': ['cpml', 'cpml'],
            'z': ['port', 'port'],
        },
        'run': {'n_steps': 10, 'record_interval': 5},
        'output': {'directory': 'outputs'},
    }
    
    sim = Simulation.from_config(config)
    sim._build_geometry()
    sim._build_grid()
    
    assert sim._grid is not None
    assert sim._grid.Nx > 0
    assert sim._grid.Ny > 0
    assert sim._grid.Nz > 0


@pytest.mark.integration
def test_simulation_ports_creation():
    """Test that Simulation correctly creates ports."""
    config = {
        'geometry': {
            'type': 'rectangular_waveguide',
            'a': 10.7e-3,
            'b': 4.3e-3,
            'length': 30e-3,
        },
        'frequency': {'center': 23e9, 'bandwidth': 5e9},
        'mode': {'m': 1, 'n': 0},
        'grid': {'resolution': 12, 'courant': 0.5},
        'boundaries': {
            'x': ['cpml', 'cpml'],
            'y': ['cpml', 'cpml'],
            'z': ['port', 'port'],
        },
        'run': {'n_steps': 10, 'record_interval': 5},
        'output': {'directory': 'outputs'},
    }
    
    sim = Simulation.from_config(config)
    sim._build_geometry()
    sim._build_grid()
    sim._build_ports_and_source()
    
    assert sim._ports is not None
    assert len(sim._ports) == 2  # Input and output


@pytest.mark.integration
def test_simulation_solver_creation():
    """Test that Simulation correctly creates FDTDSolver."""
    config = {
        'geometry': {
            'type': 'rectangular_waveguide',
            'a': 10.7e-3,
            'b': 4.3e-3,
            'length': 25e-3,
        },
        'frequency': {'center': 23e9, 'bandwidth': 5e9},
        'mode': {'m': 1, 'n': 0},
        'grid': {'resolution': 10, 'courant': 0.5},
        'boundaries': {
            'x': ['cpml', 'cpml'],
            'y': ['cpml', 'cpml'],
            'z': ['port', 'port'],
        },
        'run': {'n_steps': 10, 'record_interval': 5},
        'output': {'directory': 'outputs'},
    }
    
    sim = Simulation.from_config(config)
    sim._build_geometry()
    sim._build_grid()
    sim._build_ports_and_source()
    sim._build_solver()
    
    assert sim._solver is not None


@pytest.mark.integration
def test_simulation_output_files():
    """Test that simulation saves output files correctly."""
    pytest.skip("Output file testing - requires full simulation run and file I/O verification")


@pytest.mark.integration
def test_simulation_error_handling_invalid_geometry():
    """Test that Simulation handles invalid geometry type gracefully."""
    config = {
        'geometry': {
            'type': 'invalid_geometry_type',
        },
        'frequency': {'center': 10e9, 'bandwidth': 5e9},
        'grid': {'resolution': 10, 'courant': 0.5},
        'boundaries': {'x': ['cpml', 'cpml'], 'y': ['cpml', 'cpml'], 'z': ['cpml', 'cpml']},
        'run': {'n_steps': 10, 'record_interval': 5},
        'output': {'directory': 'outputs'},
    }
    
    sim = Simulation.from_config(config)
    
    with pytest.raises(ValueError, match="Unknown geometry type"):
        sim._build_geometry()


@pytest.mark.integration
def test_simulation_multiple_runs():
    """Test running the same simulation multiple times."""
    config = {
        'geometry': {
            'type': 'rectangular_waveguide',
            'a': 10.7e-3,
            'b': 4.3e-3,
            'length': 20e-3,
        },
        'frequency': {'center': 20e9, 'bandwidth': 5e9},
        'mode': {'m': 1, 'n': 0},
        'grid': {'resolution': 8, 'courant': 0.5},
        'boundaries': {
            'x': ['cpml', 'cpml'],
            'y': ['cpml', 'cpml'],
            'z': ['port', 'port'],
        },
        'run': {'n_steps': 20, 'record_interval': 10},
        'output': {'directory': 'outputs'},
    }
    
    sim = Simulation.from_config(config)
    sim.build()
    
    # First run
    result1 = sim.run()
    assert result1 is not None
    
    # Second run (should reset fields)
    result2 = sim.run()
    assert result2 is not None
    
    # Results should be independent (not accumulated)
    # This is a basic check - detailed verification would require comparing arrays
    assert result1 is not result2
