"""Test helper functions for common FDTD operations.

This module provides reusable utilities for test code to avoid duplication
and ensure consistent usage of the FDTD API across all tests.
"""

import numpy as np
import tensorflow as tf
from emsim.fdtd.fields import update_H, update_E
from emsim.boundaries.cpml import CPML
from emsim.boundaries.pec import apply_pec


def run_fdtd_loop(grid, n_steps, source=None, cpml=None, pml_faces=None,
                  pec_faces=None, pec_regions=None, record_interval=1,
                  injection_point=None):
    """Execute FDTD time loop with proper API.
    
    Parameters
    ----------
    grid : YeeGrid
        The computational grid with fields and materials.
    n_steps : int
        Number of time steps to run.
    source : callable, optional
        Source waveform function source(t) returning amplitude.
    cpml : CPML, optional
        Pre-configured CPML object. If None and pml_faces is provided,
        a new CPML will be created.
    pml_faces : set of str, optional
        Faces with PML (e.g., {'x-', 'x+', 'z-', 'z+'}). Only used if
        cpml is None.
    pec_faces : set of str, optional
        Faces with PEC boundary condition.
    pec_regions : list of dict, optional
        Internal PEC patches.
    record_interval : int, optional
        Record fields every N steps (default: 1).
    injection_point : tuple of int, optional
        (k, j, i) indices for source injection. If None, uses grid center.
    
    Returns
    -------
    dict
        Dictionary with final fields (Ex, Ey, Ez, Hx, Hy, Hz) and
        history dict with recorded data.
    """
    # Setup materials
    mat = grid.materials
    
    # Setup CPML if needed
    if cpml is None and pml_faces:
        cpml = CPML(
            Nz=grid.Nz, Ny=grid.Ny, Nx=grid.Nx,
            N_pml=8, dx=grid.dx, dy=grid.dy, dz=grid.dz,
            dt=grid.dt, dt_over_mu=mat.dt_over_mu, Cb=mat.Cb,
            pml_faces=pml_faces
        )
    
    # Determine injection point
    if injection_point is None:
        k_inj = grid.Nz // 2
        j_inj = grid.Ny // 2
        i_inj = grid.Nx // 2
    else:
        k_inj, j_inj, i_inj = injection_point
    
    # Curl coefficients (uniform or non-uniform grid)
    coeffs = grid.get_curl_coefficients()
    inv_dx, inv_dy, inv_dz = coeffs["inv_dx"], coeffs["inv_dy"], coeffs["inv_dz"]

    # Initialize history
    history = {
        'energy': [],
        'Ex_center': [],
        'Ey_center': [],
        'Ez_center': []
    }
    
    for n in range(n_steps):
        # Update H
        update_H(grid.Ex, grid.Ey, grid.Ez, grid.Hx, grid.Hy, grid.Hz,
                 mat.dt_over_mu, inv_dx, inv_dy, inv_dz)
        
        if cpml:
            cpml.update_H(grid.Ex, grid.Ey, grid.Ez,
                         grid.Hx, grid.Hy, grid.Hz)
        
        # Update E
        update_E(grid.Ex, grid.Ey, grid.Ez, grid.Hx, grid.Hy, grid.Hz,
                 mat.Ca, mat.Cb, inv_dx, inv_dy, inv_dz)
        
        if cpml:
            cpml.update_E(grid.Ex, grid.Ey, grid.Ez,
                         grid.Hx, grid.Hy, grid.Hz)
        
        # Source injection (Variable slice has no assign_add; use scatter update)
        if source:
            amplitude = source(n * grid.dt)
            amp = float(amplitude.numpy()) if hasattr(amplitude, 'numpy') else float(amplitude)
            idx = tf.constant([[k_inj, j_inj, i_inj]], dtype=tf.int32)
            new_val = grid.Ez[k_inj, j_inj, i_inj].numpy() + amp
            updated = tf.tensor_scatter_nd_update(
                grid.Ez.read_value(), idx, tf.constant([new_val], dtype=grid.Ez.dtype)
            )
            grid.Ez.assign(updated)
        
        # Apply PEC boundaries
        if pec_faces:
            apply_pec(grid.Ex, grid.Ey, grid.Ez, pec_faces)
        
        # Apply PEC patches
        if pec_regions:
            from emsim.boundaries.pec import apply_pec_patch
            for region in pec_regions:
                apply_pec_patch(
                    grid.Ex, grid.Ey, grid.Ez,
                    i_range=region['i_range'],
                    j_range=region['j_range'],
                    k=region['k'],
                    normal=region['normal']
                )
        
        # Record data
        if n % record_interval == 0:
            energy = compute_energy(grid)
            history['energy'].append(energy)
            history['Ex_center'].append(
                grid.Ex[grid.Nz//2, grid.Ny//2, grid.Nx//2].numpy()
            )
            history['Ey_center'].append(
                grid.Ey[grid.Nz//2, grid.Ny//2, grid.Nx//2].numpy()
            )
            history['Ez_center'].append(
                grid.Ez[grid.Nz//2, grid.Ny//2, grid.Nx//2].numpy()
            )
    
    return {
        'Ex': grid.Ex,
        'Ey': grid.Ey,
        'Ez': grid.Ez,
        'Hx': grid.Hx,
        'Hy': grid.Hy,
        'Hz': grid.Hz,
        'history': history,
        'grid': grid
    }


def compute_energy(grid):
    """Compute total electromagnetic energy in the grid.
    
    Parameters
    ----------
    grid : YeeGrid
        The computational grid with fields.
    
    Returns
    -------
    float
        Total EM energy (sum of E^2 + H^2 over all cells).
    """
    E2 = grid.Ex**2 + grid.Ey**2 + grid.Ez**2
    H2 = grid.Hx**2 + grid.Hy**2 + grid.Hz**2
    return tf.reduce_sum(E2 + H2).numpy()


def assert_energy_conservation(energy_history, tolerance=0.05, min_steps=50):
    """Assert that energy is conserved within tolerance.
    
    Parameters
    ----------
    energy_history : list of float
        History of total energy values.
    tolerance : float, optional
        Maximum relative energy variation allowed (default: 0.05 = 5%).
    min_steps : int, optional
        Minimum number of steps required in history (default: 50).
    
    Raises
    ------
    AssertionError
        If energy variation exceeds tolerance.
    """
    assert len(energy_history) >= min_steps, \
        f"Need at least {min_steps} energy samples, got {len(energy_history)}"
    
    energy_array = np.array(energy_history)
    peak_energy = np.max(energy_array)
    
    # Skip initial zeros if present
    nonzero_energy = energy_array[energy_array > peak_energy * 0.01]
    
    if len(nonzero_energy) > 0:
        energy_variation = np.std(nonzero_energy) / np.mean(nonzero_energy)
    else:
        energy_variation = 0.0
    
    assert energy_variation < tolerance, \
        f"Energy variation {energy_variation:.2%} exceeds tolerance {tolerance:.2%}"


def assert_field_magnitude_reasonable(fields, max_value=1e10):
    """Assert that field values are finite and not too large.
    
    Parameters
    ----------
    fields : list of tf.Variable
        List of field components to check.
    max_value : float, optional
        Maximum allowed absolute value (default: 1e10).
    
    Raises
    ------
    AssertionError
        If any field contains NaN, Inf, or values exceeding max_value.
    """
    for field in fields:
        assert not tf.reduce_any(tf.math.is_nan(field)).numpy(), \
            "Field contains NaN values"
        assert not tf.reduce_any(tf.math.is_inf(field)).numpy(), \
            "Field contains Inf values"
        max_field = tf.reduce_max(tf.abs(field)).numpy()
        assert max_field < max_value, \
            f"Field magnitude {max_field:.2e} exceeds limit {max_value:.2e}"


def measure_wave_speed(signal_history_1, signal_history_2, distance, dt):
    """Measure wave propagation speed using cross-correlation.

    Convention: signal_1 is recorded upstream, signal_2 downstream. The wave
    propagates in that direction, so signal_2 is a delayed copy of signal_1.
    For np.correlate(s1, s2, mode='full'), index k has lag = k - (len(s1)-1)
    in samples. Maximum correlation occurs when lag = delay_steps (positive).

    Parameters
    ----------
    signal_history_1 : array-like
        Signal at first position (upstream).
    signal_history_2 : array-like
        Signal at second position (downstream).
    distance : float
        Physical distance from position 1 to position 2 [m].
    dt : float
        Time step [s].

    Returns
    -------
    float
        Measured wave speed [m/s].
    """
    s1 = np.array(signal_history_1)
    s2 = np.array(signal_history_2)
    corr = np.correlate(s1, s2, mode='full')
    # Indices 0..len(s1)-1 correspond to negative lag; len(s1)-1..end to non-negative lag.
    # Restrict to non-negative lag (s2 delayed w.r.t. s1) and take the peak there.
    positive_lag_slice = corr[len(s1) - 1:]
    if len(positive_lag_slice) == 0:
        delay_steps = 1
    else:
        delay_steps = np.argmax(positive_lag_slice)
        if delay_steps < 1:
            delay_steps = 1
    delay_time = delay_steps * dt
    measured_speed = distance / delay_time
    return measured_speed


def setup_basic_grid(size_mm=(10, 10, 20), f0=10e9, resolution=15, **kwargs):
    """Create a basic YeeGrid for testing.
    
    Parameters
    ----------
    size_mm : tuple of float, optional
        Grid size in millimeters (x, y, z).
    f0 : float, optional
        Center frequency [Hz] (default: 10 GHz).
    resolution : int, optional
        Cells per wavelength (default: 15).
    **kwargs
        Additional parameters passed to YeeGrid (e.g., eps_r, sigma).
    
    Returns
    -------
    YeeGrid
        Configured grid ready for simulation.
    """
    from emsim.fdtd.grid import YeeGrid
    
    x_size, y_size, z_size = size_mm
    grid = YeeGrid(
        x_range=(0, x_size * 1e-3),
        y_range=(0, y_size * 1e-3),
        z_range=(0, z_size * 1e-3),
        f0=f0,
        resolution=resolution,
        courant=0.5,
        **kwargs
    )
    return grid
