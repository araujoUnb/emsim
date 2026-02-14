"""Main FDTD solver orchestrating the time loop with all components.

Integrates:
  - YeeGrid (field storage + CFL)
  - Vectorised field updates (fields.py)
  - CPML absorbing boundaries (cpml.py)
  - PEC boundaries (pec.py)
  - Gaussian pulse source injection (sources/)
  - Port recording and S-parameter extraction (ports/)
  - Support for multiple port types (modal, lumped)
  - Optional NF2FF recording for antenna simulations
"""

import math
from typing import Callable, Optional, Sequence, List

import numpy as np
import tensorflow as tf

from emsim.fdtd.grid import YeeGrid
from emsim.fdtd.fields import update_H, update_E
from emsim.fdtd.fields_dispersive import update_E_with_dispersion
from emsim.boundaries.cpml import CPML
from emsim.boundaries.pec import apply_pec, apply_pec_patch
from emsim.sources.injector import inject_soft_source
from emsim.ports.s_parameters import compute_s_parameters
from emsim.ports.port import Port
from emsim.ports.lumped_port import LumpedPort


class FDTDSolver:
    """Generalized 3D FDTD solver for waveguides, antennas, and arbitrary structures.
    
    This solver supports multiple types of ports (modal, lumped) and optional
    near-field to far-field recording, enabling simulations of waveguides,
    antennas, microstrip circuits, and more.

    Parameters
    ----------
    grid       : YeeGrid
        The computational grid with fields and materials.
    source     : callable
        Source waveform object (e.g. GaussianPulse). Must be callable as source(t).
    ports      : list of PortBase, optional
        List of ports (any type: Port, LumpedPort, etc.). For backward
        compatibility, can also pass input_port and output_port separately.
    input_port : Port, optional (deprecated)
        Legacy parameter for modal waveguide port (use ports instead).
    output_port : Port, optional (deprecated)
        Legacy parameter for modal waveguide port (use ports instead).
    Z_mode_func : callable, optional
        Function Z(f) returning mode impedance [Ohm] for modal ports.
    pec_faces  : set of str, optional
        Faces with PEC boundary condition (e.g. {'x-', 'x+', 'y-', 'y+'}).
    pml_faces  : set of str, optional
        Faces with CPML boundary condition (e.g. {'z-', 'z+'}).
    pec_regions : list of dict, optional
        List of internal PEC patches. Each dict has keys: 'i_range', 'j_range',
        'k', 'normal' for apply_pec_patch().
    n_pml      : int
        Number of PML cells (default 8).
    n_steps    : int or None
        Number of time steps. If None, computed from source duration.
    conv_tol   : float or None
        Temporal convergence tolerance (see notes). None = run all steps.
    nf2ff_box  : NF2FFBox or None
        Optional near-field to far-field recording box for antennas.
    nf2ff_record_interval : int
        Record nf2ff fields every N steps (default 10).
    
    Notes
    -----
    Temporal convergence: stops when relative field change
    ||E(t) - E(t-100dt)|| / ||E(t)|| < conv_tol. Checked every 100 steps
    after at least 25% of n_steps (minimum 500 steps) to allow source
    propagation and reflections to settle. Use ~1e-6 for waveguides.
    """

    def __init__(self, grid, source, 
                 ports: Optional[List] = None,
                 input_port=None, output_port=None, Z_mode_func=None,
                 pec_faces=None, pml_faces=None,
                 pec_regions: Optional[List[dict]] = None,
                 n_pml=8, n_steps=None, conv_tol=None,
                 nf2ff_box=None, nf2ff_record_interval=10):
        self.grid = grid
        self.source = source
        
        # Handle backward compatibility: input_port/output_port or ports list
        if ports is not None:
            self.ports = ports
        elif input_port is not None and output_port is not None:
            # Legacy mode: two modal ports
            self.ports = [input_port, output_port]
            self.input_port = input_port
            self.output_port = output_port
        else:
            self.ports = []
            self.input_port = None
            self.output_port = None
        
        self.Z_mode_func = Z_mode_func
        self.pec_faces = pec_faces or set()
        self.pml_faces = pml_faces or set()
        self.pec_regions = pec_regions or []
        self.n_pml = n_pml
        self.conv_tol = conv_tol
        self.nf2ff_box = nf2ff_box
        self.nf2ff_record_interval = nf2ff_record_interval

        # Compute number of time steps
        if n_steps is not None:
            self.n_steps = n_steps
        else:
            duration = getattr(source, 'duration', 1e-9)
            # Run for 3x source duration to capture reflections
            self.n_steps = int(math.ceil(3.0 * duration / grid.dt))

        # Initialise CPML if any PML faces are requested
        self.cpml = None
        if self.pml_faces:
            self.cpml = CPML(
                Nz=grid.Nz, Ny=grid.Ny, Nx=grid.Nx,
                N_pml=n_pml,
                dx=grid.dx, dy=grid.dy, dz=grid.dz,
                dt=grid.dt,
                dt_over_mu=grid.materials.dt_over_mu,
                Cb=grid.materials.Cb,
                pml_faces=pml_faces,
            )

        # Use combined E-update when dispersive or anisotropic regions are present
        has_drude = (
            len(grid.materials.dispersive_regions) > 0
            and getattr(grid.materials, "_drude_mask", None) is not None
        )
        has_aniso = (
            len(grid.materials.anisotropic_regions) > 0
            and getattr(grid.materials, "_aniso_mask", None) is not None
        )
        self.has_dispersive = has_drude or has_aniso

        # Precompute source waveform for all time steps.
        # Use float64 for time values to avoid precision loss at large step
        # counts (n*dt with dt ~ 1e-13 needs >7 significant digits).
        t_all = tf.cast(tf.range(self.n_steps, dtype=tf.int64), tf.float64) * float(grid.dt)
        t_all = tf.cast(t_all, tf.float32)  # source operates in float32
        self.waveform = source(t_all)  # [n_steps]

    def run(
        self,
        verbose=True,
        record_interval=1,
        snapshot_interval=0,
        checkpoint_steps: Optional[Sequence[int]] = None,
        checkpoint_callback: Optional[Callable[..., None]] = None,
    ):
        """Execute the FDTD simulation.

        Parameters
        ----------
        verbose : bool
            Print progress every 10%.
        record_interval : int
            Record port fields every N steps (default 1 = every step).
        snapshot_interval : int
            If > 0, store 2D Ez snapshots at the midplane every N steps.
        checkpoint_steps : sequence of int, optional
            Step indices at which to call checkpoint_callback (e.g. [2500, 5000, 7500, 10000]).
        checkpoint_callback : callable, optional
            Called as callback(step, incident_record, reflected_record, transmitted_record, dt)
            at each step in checkpoint_steps. Used to save S-parameter CSVs at progress points.

        Returns
        -------
        dict with keys:
            'S11', 'S21', 'freqs' : S-parameters and frequency vector
            'Ez_snapshots'        : list of 2D numpy arrays (if snapshot_interval > 0)
            'n_steps_run'         : actual number of steps executed
        """
        g = self.grid
        m = g.materials

        # Reset fields and port recordings
        g.reset_fields()
        for port in self.ports:
            port.reset()
        if self.nf2ff_box:
            self.nf2ff_box.reset()

        Ez_snapshots = []
        mid_y = g.Ny // 2
        print_every = max(self.n_steps // 10, 1)
        curl_coeffs = g.get_curl_coefficients()
        inv_dx, inv_dy, inv_dz = curl_coeffs["inv_dx"], curl_coeffs["inv_dy"], curl_coeffs["inv_dz"]
        
        # For temporal convergence check (only after minimum 25% of steps)
        min_steps_before_conv = max(self.n_steps // 4, 500)  # at least 25% or 500 steps
        Ey_prev = None
        if self.conv_tol is not None:
            Ey_prev = tf.Variable(tf.zeros_like(g.Ey), dtype=tf.float32)

        # Buffer for H temporal interpolation.  In the Yee leapfrog scheme
        # E^n and H^{n+1/2} are never co-temporal.  We store H^{n-1/2}
        # before the H-update so we can average it with H^{n+1/2} to
        # obtain H interpolated to time n, matching E^n.
        Hx_prev = tf.Variable(tf.zeros_like(g.Hx))

        for n in range(self.n_steps):
            amp = self.waveform[n]

            # --- 1. Save H^{n-1/2} for temporal interpolation ---
            Hx_prev.assign(g.Hx)

            # --- 2. H-field update  (H^{n-1/2} → H^{n+1/2}) ---
            update_H(g.Ex, g.Ey, g.Ez, g.Hx, g.Hy, g.Hz,
                     m.dt_over_mu, inv_dx, inv_dy, inv_dz)

            # --- 3. CPML H corrections ---
            if self.cpml is not None:
                self.cpml.update_H(g.Ex, g.Ey, g.Ez, g.Hx, g.Hy, g.Hz)

            # --- 4. Record E^n with H interpolated to time n ---
            # H_interp = (H^{n-1/2} + H^{n+1/2}) / 2 ≈ H^n, co-temporal
            # with E^n (which has not been updated yet this iteration).
            if n % record_interval == 0:
                Hx_interp = 0.5 * (Hx_prev + g.Hx)
                for port in self.ports:
                    if isinstance(port, Port):
                        # Modal port: record overlap integrals
                        port.record(g.Ey, Hx_interp, g.dy, g.dx)
                    elif isinstance(port, LumpedPort):
                        # Lumped port: record V and I (use local spacing if non-uniform)
                        i, j, k = port.position
                        dl = g.dz_at(k)
                        ds = g.dx_at(i) * g.dy_at(j)
                        port.record(g.Ez, g.Hx, g.Hy, dl=dl, ds=ds)

            # --- 5. E-field update  (E^n → E^{n+1}) ---
            if self.has_dispersive:
                update_E_with_dispersion(
                    g.Ex, g.Ey, g.Ez, g.Hx, g.Hy, g.Hz,
                    m, inv_dx, inv_dy, inv_dz, g.dt,
                )
            else:
                update_E(g.Ex, g.Ey, g.Ez, g.Hx, g.Hy, g.Hz,
                         m.Ca, m.Cb, inv_dx, inv_dy, inv_dz)

            # --- 6. CPML E corrections ---
            if self.cpml is not None:
                self.cpml.update_E(g.Ex, g.Ey, g.Ez, g.Hx, g.Hy, g.Hz)

            # --- 7. Source injection ---
            for port in self.ports:
                if isinstance(port, Port) and hasattr(port, 'k_plane') and hasattr(port, 'mode_profile_E'):
                    # Modal port injection
                    inject_soft_source(g.Ey, port.k_plane, port.mode_profile_E, amp)
                elif isinstance(port, LumpedPort):
                    # Lumped port injection (use local dz at port)
                    _, _, k = port.position
                    port.inject(g.Ez, amp, dl=g.dz_at(k))

            # --- 8. PEC boundary enforcement (must be last field modification) ---
            if self.pec_faces:
                apply_pec(g.Ex, g.Ey, g.Ez, self.pec_faces)
            
            # --- 8b. PEC patches (internal conductors) ---
            for pec_region in self.pec_regions:
                apply_pec_patch(g.Ex, g.Ey, g.Ez, **pec_region)

            # --- 9. NF2FF recording (for antennas) ---
            if self.nf2ff_box and n % self.nf2ff_record_interval == 0:
                self.nf2ff_box.record(g.Ex, g.Ey, g.Ez, g.Hx, g.Hy, g.Hz)

            # --- 10. Ez snapshot ---
            if snapshot_interval > 0 and n % snapshot_interval == 0:
                Ez_snapshots.append(g.Ez[:, mid_y, :].numpy().copy())

            # --- 11. Progress ---
            if verbose and (n % print_every == 0):
                max_e = float(tf.reduce_max(tf.abs(g.Ey)).numpy())
                print(f"  Step {n:6d}/{self.n_steps}  |Ey|_max = {max_e:.4e}")

            # --- 12. Temporal convergence check ---
            if self.conv_tol is not None and n >= min_steps_before_conv and n % 100 == 0:
                diff = tf.abs(g.Ey - Ey_prev)
                max_diff = float(tf.reduce_max(diff).numpy())
                max_field = float(tf.reduce_max(tf.abs(g.Ey)).numpy())

                if max_field > 1e-12:
                    relative_change = max_diff / max_field
                    if relative_change < self.conv_tol:
                        if verbose:
                            print(f"  Converged at step {n} (relative change = {relative_change:.2e})")
                            print(f"  (minimum {min_steps_before_conv} steps enforced)")
                        break

                Ey_prev.assign(g.Ey)

        # --- Post-processing: compute results from each port ---
        result = {}
        
        # Handle different port types
        if len(self.ports) == 2 and all(isinstance(p, Port) for p in self.ports):
            # Legacy mode: two modal ports for S-parameters
            result = compute_s_parameters(
                E_record_input=self.ports[0].E_record,
                H_record_input=self.ports[0].H_record,
                E_record_output=self.ports[1].E_record,
                H_record_output=self.ports[1].H_record,
                Z_mode_func=self.Z_mode_func,
                dt=g.dt * record_interval,
            )
        else:
            # Generalized mode: each port computes its own result
            for port in self.ports:
                port_result = port.compute_result(dt=g.dt * record_interval)
                
                if port_result['type'] == 'modal':
                    # For modal ports, we may still need S-parameters
                    # Store the records for later processing
                    result[f'{port.name}_E_record'] = port_result['E_record']
                    result[f'{port.name}_H_record'] = port_result['H_record']
                
                elif port_result['type'] == 'lumped':
                    # Lumped port provides Z_in and S11 directly
                    result['freqs'] = port_result['freqs']
                    result['Z_in'] = port_result['Z_in']
                    result['S11'] = port_result['S11']
                    result['V_record'] = port_result['V_record']
                    result['I_record'] = port_result['I_record']
        
        result['Ez_snapshots'] = Ez_snapshots
        result['n_steps_run'] = n + 1
        
        # Add nf2ff box if present (will be processed separately)
        if self.nf2ff_box:
            result['nf2ff_box'] = self.nf2ff_box

        return result
