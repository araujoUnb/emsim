# emsim

Electromagnetic simulations (3D FDTD with TensorFlow).

## Features

- Yee-grid FDTD with CPML and PEC boundaries
- Materials: isotropic catalog, dispersive (Drude, Lorentz, Debye), anisotropic (diagonal tensor)
- Ports: modal waveguide ports and lumped ports for S-parameters and impedance
- Patch antenna and rectangular waveguide geometries

## Documentation

- **[Materials](docs/materials.md)**: Catalog, manager, CSV/JSON libraries, dispersive and anisotropic materials, YAML config.
- **[Performance](docs/performance.md)**: Benchmarks, cells/second, device setup, XLA and mixed precision.
