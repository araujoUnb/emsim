"""High-level simulation interface with YAML-driven configuration.

This module provides a single entry point for running FDTD simulations:
load a YAML config, build the grid/solver, run. Results and checkpoints
are saved to the output dir; use a separate postprocess script to generate
figures from the saved result.

Example
-------
    sim = Simulation.from_yaml("Simulations/WR42/config.yaml")
    result = sim.run()
    # Then: python Simulations/WR42/postprocess.py
"""

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from emsim.constants import C0
from emsim.device import setup_device
from emsim.fdtd.grid import YeeGrid
from emsim.fdtd.solver import FDTDSolver
from emsim.geometry.rectangular_waveguide import RectangularWaveguide
from emsim.modes.rectangular import te_mode_profile, hx_mode_profile, cutoff_frequency, mode_impedance
from emsim.ports.port import Port
from emsim.sources.gaussian_pulse import GaussianPulse


def _parse_mode(mode_str: str) -> tuple:
    """Parse mode string (e.g. 'TE10', 'TM21') into (m, n) indices.

    Parameters
    ----------
    mode_str : str
        Mode identifier such as 'TE10' or 'TM21'.

    Returns
    -------
    tuple of (int, int)
        (m, n) mode indices.

    Raises
    ------
    ValueError
        If mode_str does not match pattern T[EM]mn.
    """
    m = re.match(r"T[EM](\d+)(\d+)", mode_str.strip().upper())
    if not m:
        raise ValueError(f"Invalid mode string: {mode_str!r}. Expected e.g. 'TE10', 'TM21'.")
    return int(m.group(1)), int(m.group(2))


def _apply_overrides(config: dict, overrides: Dict[str, Any]) -> None:
    """Apply dot-notation overrides into a nested config dict (in-place).

    Parameters
    ----------
    config : dict
        Configuration dictionary to modify.
    overrides : dict
        Keys are dot-separated paths (e.g. 'grid.resolution'), values are
        the new values to set.
    """
    for key, value in overrides.items():
        parts = key.split(".")
        d = config
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = value


class Simulation:
    """High-level FDTD simulation interface driven by a YAML configuration.

    The workflow is: load config (from YAML) → build (grid, ports, solver)
    → run. The simulation box (domain) and all physical and numerical
    parameters are defined in the YAML file. Results and checkpoints are
    saved to the output dir; use a separate postprocess script for figures.

    Attributes
    ----------
    config : dict
        The loaded configuration (after load_yaml). Read-only for users.
    grid : YeeGrid or None
        The 3D Yee grid, set after build().
    solver : FDTDSolver or None
        The FDTD solver, set after build().
    result : dict or None
        The result of run() (freqs, S11, S21, Ez_snapshots, etc.).
    """

    def __init__(self) -> None:
        """Create an empty simulation. Call load_yaml() then build() and run()."""
        self._config: Dict[str, Any] = {}
        self._config_path: Optional[str] = None
        self._geometry: Optional[RectangularWaveguide] = None
        self._grid: Optional[YeeGrid] = None
        self._solver: Optional[FDTDSolver] = None
        self._source: Optional[GaussianPulse] = None
        self._input_port: Optional[Port] = None
        self._output_port: Optional[Port] = None
        self._result: Optional[Dict[str, Any]] = None
        self._device_used: Optional[str] = None

    # -------------------------------------------------------------------------
    # Public read-only access to built objects
    # -------------------------------------------------------------------------

    @property
    def config(self) -> Dict[str, Any]:
        """Loaded YAML configuration (read-only)."""
        return self._config

    @property
    def grid(self) -> Optional[YeeGrid]:
        """Yee grid (available after build())."""
        return self._grid

    @property
    def solver(self) -> Optional[FDTDSolver]:
        """FDTD solver (available after build())."""
        return self._solver

    @property
    def result(self) -> Optional[Dict[str, Any]]:
        """Result dict from run() (freqs, S11, S21, Ez_snapshots, etc.)."""
        return self._result

    @property
    def geometry(self) -> Optional[RectangularWaveguide]:
        """Geometry object (e.g. RectangularWaveguide) after build()."""
        return self._geometry

    # -------------------------------------------------------------------------
    # Load configuration
    # -------------------------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        path: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "Simulation":
        """Create a Simulation and load its configuration from a YAML file.

        Parameters
        ----------
        path : str
            Path to the YAML configuration file.
        overrides : dict, optional
            Dot-notation overrides to apply after loading (e.g.
            {'grid.resolution': 30, 'run.n_steps': 8000}).

        Returns
        -------
        Simulation
            Instance with config loaded. Call build() then run().
        """
        sim = cls()
        sim.load_yaml(path, overrides=overrides)
        return sim

    def load_yaml(
        self,
        path: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Load simulation parameters from a YAML file.

        The file should define at least: domain, frequency, mode, grid,
        boundaries, run. See Simulations/WR42/config.yaml for a full example.

        Parameters
        ----------
        path : str
            Path to the YAML file.
        overrides : dict, optional
            Dot-notation overrides (e.g. {'grid.n_pml': 6}).
        """
        path_obj = Path(path)
        if not path_obj.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        self._config_path = str(path_obj.resolve())
        with open(path_obj, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)
        if self._config is None:
            self._config = {}
        if overrides:
            _apply_overrides(self._config, overrides)

    # -------------------------------------------------------------------------
    # Build: construct grid, ports, source, solver from config
    # -------------------------------------------------------------------------

    def build(self) -> None:
        """Build geometry, grid, ports, source and solver from the loaded config.

        Must be called after load_yaml(). Uses the following config sections:
        domain, frequency, mode, grid, boundaries, run.

        Raises
        ------
        ValueError
            If config is missing required keys or mode string is invalid.
        """
        device_choice = self._config.get("device", "auto")
        self._device_used = setup_device(device_choice)
        self._build_geometry()
        self._build_grid()
        self._build_ports_and_source()
        self._build_solver()

    def _build_geometry(self) -> None:
        """Build geometry from config (supports multiple types)."""
        geom_cfg = self._config.get("geometry", {})
        geom_type = geom_cfg.get("type", "rectangular_waveguide")
        
        if geom_type == "rectangular_waveguide":
            # Legacy mode: use 'domain' for waveguide dimensions
            if "domain" not in self._config:
                raise ValueError("config must contain 'domain' with x, y, z ranges.")
            dom = self._config["domain"]
            x0, x1 = float(dom["x"][0]), float(dom["x"][1])
            y0, y1 = float(dom["y"][0]), float(dom["y"][1])
            z0, z1 = float(dom["z"][0]), float(dom["z"][1])
            a = x1 - x0
            b = y1 - y0
            length = z1 - z0
            self._geometry = RectangularWaveguide(a=a, b=b, length=length)
        
        elif geom_type == "patch_antenna":
            # New mode: patch antenna with substrate
            from emsim.geometry.patch_antenna import PatchAntenna
            g = geom_cfg
            sub = g["substrate"]
            self._geometry = PatchAntenna(
                patch_width=float(g["patch_width"]),
                patch_length=float(g["patch_length"]),
                substrate_width=float(sub["width"]),
                substrate_length=float(sub["length"]),
                substrate_thickness=float(sub["thickness"]),
                substrate_eps_r=float(sub["eps_r"]),
                substrate_kappa=float(sub.get("kappa", 0)),
                feed_x=float(g["feed_x"]),
                sim_box=tuple(float(x) for x in g["sim_box"])
            )
        
        else:
            raise ValueError(f"Unknown geometry type: {geom_type}. "
                           f"Supported: 'rectangular_waveguide', 'patch_antenna'.")

    def _build_grid(self) -> None:
        """Build YeeGrid from config (domain, frequency, grid)."""
        from emsim.geometry.patch_antenna import PatchAntenna
        
        freq = self._config["frequency"]
        gr = self._config["grid"]
        n_pml = int(gr["n_pml"])
        f0 = float(freq["f0"])
        resolution = int(gr["resolution"])
        
        # Get spatial ranges from geometry
        x0, x1 = self._geometry.x_range
        y0, y1 = self._geometry.y_range
        z0, z1 = self._geometry.z_range
        
        # Add PML padding
        dz_approx = (C0 / f0) / resolution
        pml_pad = n_pml * dz_approx
        z_min = z0 - pml_pad
        z_max = z1 + pml_pad
        
        self._grid = YeeGrid(
            x_range=(x0, x1),
            y_range=(y0, y1),
            z_range=(z_min, z_max),
            f0=f0,
            resolution=resolution,
            courant=float(gr["courant"]),
        )
        
        # For patch antenna: set substrate material region
        if isinstance(self._geometry, PatchAntenna):
            # Calculate grid indices for substrate region
            # Substrate goes from z=0 to z=thickness (ground plane at z=0)
            k_substrate = int(self._geometry.substrate_thickness / self._grid.dz)
            
            # Substrate occupies central region in x and y
            i_sub_min = (self._grid.Nx - int(self._geometry.substrate_width / self._grid.dx)) // 2
            i_sub_max = i_sub_min + int(self._geometry.substrate_width / self._grid.dx)
            j_sub_min = (self._grid.Ny - int(self._geometry.substrate_length / self._grid.dy)) // 2
            j_sub_max = j_sub_min + int(self._geometry.substrate_length / self._grid.dy)
            
            # Ground is at k=n_pml (first cell after PML)
            k_ground = n_pml
            
            # Set substrate material properties
            self._grid.materials.set_region(
                i_range=(i_sub_min, i_sub_max),
                j_range=(j_sub_min, j_sub_max),
                k_range=(k_ground, k_ground + k_substrate),
                eps_r=self._geometry.substrate_eps_r,
                sigma=self._geometry.substrate_kappa
            )
            
            # Recompute coefficients after material change
            self._grid.materials.compute_coefficients(self._grid.dt)

    def _build_ports_and_source(self) -> None:
        """Build ports and Gaussian pulse source from config."""
        from emsim.geometry.patch_antenna import PatchAntenna
        from emsim.ports.lumped_port import LumpedPort
        
        freq = self._config["frequency"]
        n_pml = int(self._config["grid"]["n_pml"])
        
        # Check if we have a port config (patch antenna) or mode config (waveguide)
        port_cfg = self._config.get("port", {})
        mode_cfg = self._config.get("mode", {})
        
        if port_cfg and port_cfg.get("type") == "lumped":
            # Patch antenna with lumped port
            if not isinstance(self._geometry, PatchAntenna):
                raise ValueError("Lumped port requires PatchAntenna geometry")
            
            geom = self._geometry
            
            # Calculate feed position in grid coordinates
            # Feed is at the specified x position, center in y, on top of substrate
            feed_x_abs = geom.feed_x  # relative to center
            x_center = (geom.x_range[0] + geom.x_range[1]) / 2
            feed_x_world = x_center + feed_x_abs
            
            i_feed = int((feed_x_world - geom.x_range[0]) / self._grid.dx)
            j_feed = self._grid.Ny // 2
            k_substrate = int(geom.substrate_thickness / self._grid.dz)
            k_feed = n_pml + k_substrate  # top of substrate
            
            port = LumpedPort(
                name="feed",
                i=i_feed, j=j_feed, k=k_feed,
                direction='z',
                resistance=float(port_cfg.get("resistance", 50.0))
            )
            self._ports = [port]
            self._Z_mode_func = None  # Not used for lumped ports
        
        elif mode_cfg and mode_cfg.get("type"):
            # Waveguide with modal ports
            m, n = _parse_mode(mode_cfg["type"])
            
            # Generate E-field and H-field mode profiles at Yee stagger positions
            mode_E = te_mode_profile(
                m, n,
                self._geometry.a, self._geometry.b,
                self._grid.Ny, self._grid.Nx,
                dx=self._grid.dx, dy=self._grid.dy,
            )
            mode_H = hx_mode_profile(
                m, n,
                self._geometry.a, self._geometry.b,
                self._grid.Ny, self._grid.Nx,
                dx=self._grid.dx, dy=self._grid.dy,
            )
            
            k_src = n_pml + 2
            k_rec = self._grid.Nz - n_pml - 3
            
            # Create mode impedance function for S-parameter calculation
            def Z_mode_func(f: float) -> float:
                return mode_impedance(m, n, self._geometry.a, self._geometry.b, f)
            self._Z_mode_func = Z_mode_func
            
            self._input_port = Port(
                "input", k_plane=k_src, mode_profile_E=mode_E, mode_profile_H=mode_H, direction=1
            )
            self._output_port = Port(
                "output", k_plane=k_rec, mode_profile_E=mode_E, mode_profile_H=mode_H, direction=1
            )
            self._ports = [self._input_port, self._output_port]
        
        else:
            raise ValueError("Config must contain either 'port' (for lumped) or 'mode' (for waveguide)")
        
        # Source is always GaussianPulse
        self._source = GaussianPulse(
            f0=float(freq["f0"]),
            bandwidth=float(freq["bandwidth"]),
        )

    def _build_solver(self) -> None:
        """Build FDTDSolver from config and already-built grid/ports/source."""
        from emsim.geometry.patch_antenna import PatchAntenna
        from emsim.postprocessing.nf2ff import NF2FFBox
        
        run_cfg = self._config["run"]
        b = self._config["boundaries"]
        n_pml = int(self._config["grid"]["n_pml"])
        
        conv_tol = run_cfg.get("conv_tol")
        if conv_tol is not None:
            conv_tol = float(conv_tol)
        
        # Build NF2FF box if enabled (for antennas)
        nf2ff_cfg = self._config.get("nf2ff", {})
        nf2ff_box = None
        if nf2ff_cfg.get("enabled", False):
            margin_cells = int(nf2ff_cfg.get("box_margin", 10e-3) / self._grid.dx)
            i_min = max(margin_cells, 0)
            i_max = min(self._grid.Nx - margin_cells, self._grid.Nx)
            j_min = max(margin_cells, 0)
            j_max = min(self._grid.Ny - margin_cells, self._grid.Ny)
            k_min = max(margin_cells, 0)
            k_max = min(self._grid.Nz - margin_cells, self._grid.Nz)
            
            nf2ff_box = NF2FFBox(
                i_range=(i_min, i_max),
                j_range=(j_min, j_max),
                k_range=(k_min, k_max)
            )
        
        # Build PEC regions for patch antenna
        pec_regions = []
        if isinstance(self._geometry, PatchAntenna):
            geom = self._geometry
            
            # Calculate grid indices
            i_sub_min = (self._grid.Nx - int(geom.substrate_width / self._grid.dx)) // 2
            i_sub_max = i_sub_min + int(geom.substrate_width / self._grid.dx)
            j_sub_min = (self._grid.Ny - int(geom.substrate_length / self._grid.dy)) // 2
            j_sub_max = j_sub_min + int(geom.substrate_length / self._grid.dy)
            
            i_patch_min = (self._grid.Nx - int(geom.patch_width / self._grid.dx)) // 2
            i_patch_max = i_patch_min + int(geom.patch_width / self._grid.dx)
            j_patch_min = (self._grid.Ny - int(geom.patch_length / self._grid.dy)) // 2
            j_patch_max = j_patch_min + int(geom.patch_length / self._grid.dy)
            
            k_ground = n_pml
            k_substrate = int(geom.substrate_thickness / self._grid.dz)
            k_patch = k_ground + k_substrate
            
            # Ground plane
            pec_regions.append({
                'i_range': (i_sub_min, i_sub_max),
                'j_range': (j_sub_min, j_sub_max),
                'k': k_ground,
                'normal': 'z'
            })
            
            # Patch
            pec_regions.append({
                'i_range': (i_patch_min, i_patch_max),
                'j_range': (j_patch_min, j_patch_max),
                'k': k_patch,
                'normal': 'z'
            })
        
        # Build solver with generalized parameters
        self._solver = FDTDSolver(
            grid=self._grid,
            source=self._source,
            ports=self._ports,
            input_port=getattr(self, '_input_port', None),  # backward compat
            output_port=getattr(self, '_output_port', None),  # backward compat
            Z_mode_func=self._Z_mode_func,
            pec_faces=set(b.get("pec", [])),
            pml_faces=set(b["pml"]),
            pec_regions=pec_regions,
            n_pml=n_pml,
            n_steps=int(run_cfg["n_steps"]),
            conv_tol=conv_tol,
            nf2ff_box=nf2ff_box,
            nf2ff_record_interval=int(nf2ff_cfg.get("record_interval", 10))
        )

    def _format_config_summary(self) -> str:
        """Return a human-readable summary of the current config for logging/print."""
        c = self._config
        lines = [
            "--- Simulation configuration ---",
            f"  name: {c.get('name', 'N/A')}",
            f"  description: {c.get('description', 'N/A')}",
            "  domain:",
            f"    x: {c.get('domain', {}).get('x')}",
            f"    y: {c.get('domain', {}).get('y')}",
            f"    z: {c.get('domain', {}).get('z')}",
            "  frequency:",
            f"    f0: {c.get('frequency', {}).get('f0')}",
            f"    bandwidth: {c.get('frequency', {}).get('bandwidth')}",
            f"    f_min: {c.get('frequency', {}).get('f_min')}",
            f"    f_max: {c.get('frequency', {}).get('f_max')}",
            "  mode:",
            f"    type: {c.get('mode', {}).get('type')}",
            "  grid:",
            f"    resolution: {c.get('grid', {}).get('resolution')}",
            f"    n_pml: {c.get('grid', {}).get('n_pml')}",
            f"    courant: {c.get('grid', {}).get('courant')}",
            "  run:",
            f"    n_steps: {c.get('run', {}).get('n_steps')}",
            f"    snapshot_interval: {c.get('run', {}).get('snapshot_interval')}",
            f"    conv_tol: {c.get('run', {}).get('conv_tol')}",
            "  output:",
            f"    dir: {c.get('output', {}).get('dir')}",
            f"    save: {c.get('output', {}).get('save')}",
        ]
        if self._config_path:
            lines.insert(1, f"  config_file: {self._config_path}")
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Run
    # -------------------------------------------------------------------------

    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """Run the FDTD simulation.

        Prints config summary, runs the solver, saves result and requested outputs
        (from output.save) to output dir, and writes a simulation log (run_info.yaml).
        If build() has not been called, it is called automatically.

        Parameters
        ----------
        verbose : bool, optional
            Print progress every 10% of steps (default True).

        Returns
        -------
        dict
            Keys: 'freqs', 'S11', 'S21', 'Ez_snapshots', 'n_steps_run'.
        """
        if self._solver is None:
            self.build()

        run_cfg = self._config["run"]
        out_cfg = self._config.get("output", {})
        output_dir = Path(out_cfg.get("dir", "."))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Print configuration and device (device always shown)
        print(f"Using device: {self._device_used}")
        if verbose:
            print(self._format_config_summary())
            print("---")

        freq_cfg = self._config.get("frequency", {})
        f_min = freq_cfg.get("f_min")
        f_max = freq_cfg.get("f_max")
        if f_min is not None:
            f_min = float(f_min)
        if f_max is not None:
            f_max = float(f_max)

        n_steps = self._solver.n_steps

        # Timing
        record_time = out_cfg.get("record_simulation_time", True)
        t_start = time.perf_counter()
        start_time_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # Run solver
        snapshot_interval = run_cfg.get("snapshot_interval", 0)
        self._result = self._solver.run(
            verbose=verbose,
            record_interval=1,
            snapshot_interval=snapshot_interval,
        )

        t_end = time.perf_counter()
        duration_s = t_end - t_start
        end_time_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        if verbose and record_time:
            print(f"  Simulation time: {duration_s:.2f} s")

        # Save result as CSV for post-processing (no pickle)
        import numpy as np
        import pandas as pd

        saved_files: List[str] = []

        # Metadata (n_steps_run, etc.)
        metadata_path = output_dir / "result_metadata.csv"
        pd.DataFrame([{"n_steps_run": self._result.get("n_steps_run")}]).to_csv(
            metadata_path, index=False
        )
        saved_files.append("result_metadata.csv")

        # S-parameters (always saved so postprocess can plot)
        from emsim.postprocessing.s_parameters_plot import save_s_parameters_csv
        save_s_parameters_csv(
            self._result,
            str(output_dir / "s_parameters.csv"),
            f_min=f_min,
            f_max=f_max,
        )
        saved_files.append("s_parameters.csv")

        # Optional outputs from output.save (Ez snapshots as CSV)
        save_list = out_cfg.get("save") or ["s_parameters"]
        if isinstance(save_list, str):
            save_list = [save_list]
        for key in save_list:
            name = (key if isinstance(key, str) else str(key)).strip().lower()
            if name == "e":
                ez_list = self._result.get("Ez_snapshots") or []
                if ez_list:
                    rows = []
                    for idx, snap in enumerate(ez_list):
                        arr = np.asarray(snap)
                        for (i, k), val in np.ndenumerate(arr):
                            rows.append({"snapshot_index": idx, "i": i, "k": k, "Ez": val})
                    pd.DataFrame(rows).to_csv(
                        output_dir / "ez_snapshots.csv", index=False
                    )
                    saved_files.append("ez_snapshots.csv")
                break

        # Simulation log (run_info.yaml)
        log_data: Dict[str, Any] = {
            "config_file": self._config_path,
            "config_summary": {
                "name": self._config.get("name"),
                "domain": self._config.get("domain"),
                "frequency": self._config.get("frequency"),
                "mode": self._config.get("mode"),
                "grid": self._config.get("grid"),
                "run": self._config.get("run"),
                "output": self._config.get("output"),
            },
            "start_time": start_time_iso,
            "end_time": end_time_iso,
            "n_steps_planned": n_steps,
            "n_steps_run": self._result.get("n_steps_run"),
            "saved_files": saved_files,
        }
        if record_time:
            log_data["duration_s"] = round(duration_s, 2)
        log_path = output_dir / "run_info.yaml"
        with open(log_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(log_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        return self._result
