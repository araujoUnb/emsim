# tests/test_boundary_conditions.py

import torch
import pytest
import math
import numpy as np

from emsim.boundary_conditions import PECBoundary, PMLBoundary


def test_pec_boundary_on_magnetic_fields():
    """
    Verify that applying a PEC boundary to H fields zeros out the correct tangential components.
    We'll create small Hx, Hy, Hz tensors filled with ones, apply PEC on various faces,
    and check that the specified slices become zero while others remain unchanged.
    """
    # Create a small 3x3x3 grid for Hx,Hy,Hz for simplicity.
    Nz, Ny, Nx = 3, 3, 3
    Hx = torch.ones((Nz + 1, Ny, Nx))
    Hy = torch.ones((Nz, Ny + 1, Nx))
    Hz = torch.ones((Nz, Ny, Nx + 1))

    # Test face "X-": should zero Hx[:,:,0] and Hz[:,:,0]
    pec_xm = PECBoundary("X-")
    pec_xm.apply_on_H(Hx, Hy, Hz)

    # Check Hx[:,:,0] == 0, Hz[:,:,0] == 0
    assert torch.all(Hx[:, :, 0] == 0.0)
    assert torch.all(Hz[:, :, 0] == 0.0)
    # Other indices remain 1
    assert torch.all(Hx[:, :, 1:] == 1.0)
    assert torch.all(Hz[:, :, 1:] == 1.0)
    # Hy should be untouched
    assert torch.all(Hy == 1.0)

    # Reset Hx,Hy,Hz
    Hx.fill_(1.0)
    Hy.fill_(1.0)
    Hz.fill_(1.0)

    # Test face "X+": should zero Hx[:,:,-1] and Hz[:,:,-1]
    pec_xp = PECBoundary("X+")
    pec_xp.apply_on_H(Hx, Hy, Hz)
    assert torch.all(Hx[:, :, -1] == 0.0)
    assert torch.all(Hz[:, :, -1] == 0.0)
    # Check other indices are still 1
    assert torch.all(Hx[:, :, :-1] == 1.0)
    assert torch.all(Hz[:, :, :-1] == 1.0)
    assert torch.all(Hy == 1.0)

    # Reset Hx,Hy,Hz
    Hx.fill_(1.0)
    Hy.fill_(1.0)
    Hz.fill_(1.0)

    # Test face "Y-": should zero Hx[:,0,:] and Hy[:,0,:]
    pec_ym = PECBoundary("Y-")
    pec_ym.apply_on_H(Hx, Hy, Hz)
    assert torch.all(Hx[:, 0, :] == 0.0)
    assert torch.all(Hy[:, 0, :] == 0.0)
    # Other indices remain 1
    assert torch.all(Hx[:, 1:, :] == 1.0)
    assert torch.all(Hy[:, 1:, :] == 1.0)
    assert torch.all(Hz == 1.0)

    # Reset Hx,Hy,Hz
    Hx.fill_(1.0)
    Hy.fill_(1.0)
    Hz.fill_(1.0)

    # Test face "Y+": should zero Hx[:,-1,:] and Hy[:,-1,:]
    pec_yp = PECBoundary("Y+")
    pec_yp.apply_on_H(Hx, Hy, Hz)
    assert torch.all(Hx[:, -1, :] == 0.0)
    assert torch.all(Hy[:, -1, :] == 0.0)
    assert torch.all(Hx[:, :-1, :] == 1.0)
    assert torch.all(Hy[:, :-1, :] == 1.0)
    assert torch.all(Hz == 1.0)

    # Reset Hx,Hy,Hz
    Hx.fill_(1.0)
    Hy.fill_(1.0)
    Hz.fill_(1.0)

    # Test face "Z-": should zero Hy[0,:,:] and Hz[0,:,:]
    pec_zm = PECBoundary("Z-")
    pec_zm.apply_on_H(Hx, Hy, Hz)
    assert torch.all(Hy[0, :, :] == 0.0)
    assert torch.all(Hz[0, :, :] == 0.0)
    assert torch.all(Hy[1:, :, :] == 1.0)
    assert torch.all(Hz[1:, :, :] == 1.0)
    assert torch.all(Hx == 1.0)

    # Reset Hx,Hy,Hz
    Hx.fill_(1.0)
    Hy.fill_(1.0)
    Hz.fill_(1.0)

    # Test face "Z+": should zero Hy[-1,:,:] and Hz[-1,:,:]
    pec_zp = PECBoundary("Z+")
    pec_zp.apply_on_H(Hx, Hy, Hz)
    assert torch.all(Hy[-1, :, :] == 0.0)
    assert torch.all(Hz[-1, :, :] == 0.0)
    assert torch.all(Hy[:-1, :, :] == 1.0)
    assert torch.all(Hz[:-1, :, :] == 1.0)
    assert torch.all(Hx == 1.0)


def test_pec_boundary_on_electric_fields():
    """
    Verify that applying a PEC boundary to E fields zeros out the correct tangential components.
    We'll create small Ex, Ey, Ez tensors filled with ones, apply PEC on various faces,
    and check that the specified slices become zero while others remain unchanged.
    """
    # Use 3x3x3 grid for Ex, Ey, Ez for simplicity.
    Nz, Ny, Nx = 3, 3, 3
    Ex = torch.ones((Nz, Ny + 1, Nx + 1))
    Ey = torch.ones((Nz + 1, Ny, Nx + 1))
    Ez = torch.ones((Nz + 1, Ny + 1, Nx))

    # Face "X-": zero Ez[:,:,0] and Ey[:,:,0]
    pec_xm = PECBoundary("X-")
    pec_xm.apply_on_E(Ex, Ey, Ez)
    assert torch.all(Ez[:, :, 0] == 0.0)
    assert torch.all(Ey[:, :, 0] == 0.0)
    assert torch.all(Ez[:, :, 1:] == 1.0)
    # Ey at i>0 remains 1
    assert torch.all(Ey[:, :, 1:] == 1.0)
    # Ex is untouched
    assert torch.all(Ex == 1.0)

    # Reset
    Ex.fill_(1.0)
    Ey.fill_(1.0)
    Ez.fill_(1.0)

    # Face "X+": zero Ez[:,:,-1] and Ey[:,:,-1]
    pec_xp = PECBoundary("X+")
    pec_xp.apply_on_E(Ex, Ey, Ez)
    assert torch.all(Ez[:, :, -1] == 0.0)
    assert torch.all(Ey[:, :, -1] == 0.0)
    assert torch.all(Ez[:, :, :-1] == 1.0)
    assert torch.all(Ey[:, :, :-1] == 1.0)
    assert torch.all(Ex == 1.0)

    # Reset
    Ex.fill_(1.0)
    Ey.fill_(1.0)
    Ez.fill_(1.0)

    # Face "Y-": zero Ez[:,0,:] and Ex[:,0,:]
    pec_ym = PECBoundary("Y-")
    pec_ym.apply_on_E(Ex, Ey, Ez)
    assert torch.all(Ez[:, 0, :] == 0.0)
    assert torch.all(Ex[:, 0, :] == 0.0)
    assert torch.all(Ez[:, 1:, :] == 1.0)
    assert torch.all(Ex[:, 1:, :] == 1.0)
    assert torch.all(Ey == 1.0)

    # Reset
    Ex.fill_(1.0)
    Ey.fill_(1.0)
    Ez.fill_(1.0)

    # Face "Y+": zero Ez[:,-1,:] and Ex[:,-1,:]
    pec_yp = PECBoundary("Y+")
    pec_yp.apply_on_E(Ex, Ey, Ez)
    assert torch.all(Ez[:, -1, :] == 0.0)
    assert torch.all(Ex[:, -1, :] == 0.0)
    assert torch.all(Ez[:, :-1, :] == 1.0)
    assert torch.all(Ex[:, :-1, :] == 1.0)
    assert torch.all(Ey == 1.0)

    # Reset
    Ex.fill_(1.0)
    Ey.fill_(1.0)
    Ez.fill_(1.0)

    # Face "Z-": zero Ex[0,:,:] and Ey[0,:,:]
    pec_zm = PECBoundary("Z-")
    pec_zm.apply_on_E(Ex, Ey, Ez)
    assert torch.all(Ex[0, :, :] == 0.0)
    assert torch.all(Ey[0, :, :] == 0.0)
    assert torch.all(Ex[1:, :, :] == 1.0)
    assert torch.all(Ey[1:, :, :] == 1.0)
    assert torch.all(Ez == 1.0)

    # Reset
    Ex.fill_(1.0)
    Ey.fill_(1.0)
    Ez.fill_(1.0)

    # Face "Z+": zero Ex[-1,:,:] and Ey[-1,:,:]
    pec_zp = PECBoundary("Z+")
    pec_zp.apply_on_E(Ex, Ey, Ez)
    assert torch.all(Ex[-1, :, :] == 0.0)
    assert torch.all(Ey[-1, :, :] == 0.0)
    assert torch.all(Ex[:-1, :, :] == 1.0)
    assert torch.all(Ey[:-1, :, :] == 1.0)
    assert torch.all(Ez == 1.0)


def test_pml_boundary_allocation_and_profile():
    """
    Verify that PMLBoundary constructs the sigma_profile correctly and allocates
    split‐field tensors of the correct shape.
    """
    face = "X-"
    dt = 1e-12
    dx = 1e-3
    N_pml = 8
    m = 3
    R_inf = 1e-6

    pml = PMLBoundary(face, dt=dt, dx=dx, N_pml=N_pml, m=m, R_inf=R_inf)

    # Check sigma_profile length and monotonic increase
    assert pml.sigma_profile.shape == (N_pml,)
    # sigma_profile should be >0 and increasing
    sig_np = pml.sigma_profile.numpy()
    assert np.all(sig_np > 0.0)
    assert np.all(np.diff(sig_np) >= 0.0)

    # Allocate tensors for a small domain
    Nz, Ny, Nx = 5, 5, 5
    device = "cpu"
    pml.allocate_tensors(Nz, Ny, Nx, device)

    # Check shapes of split‐field tensors
    assert pml.H_xy.shape == (Nz + 1, Ny, Nx)
    assert pml.H_xz.shape == (Nz + 1, Ny, Nx)
    assert pml.E_yx.shape == (Nz, Ny + 1, Nx + 1)
    assert pml.E_zx.shape == (Nz, Ny + 1, Nx + 1)
    assert pml.H_xy.device.type == device
    assert pml.E_yx.device.type == device


def test_pml_boundary_simple_H_update():
    """
    Perform a very basic smoke test of apply_on_H for the PMLBoundary on face X-.
    We initialize Hx with ones in the PML region and Ey, Ez with arbitrary patterns,
    then call apply_on_H and check that Hx is no longer all ones in the PML. This
    verifies the method executes without error and modifies Hx in the intended region.
    """
    # Build a small domain 6×6×6 to have PML thickness=2 for this test
    Nz, Ny, Nx = 6, 6, 6
    dt = 1e-12
    dx = dy = dz = 1e-3
    N_pml = 2  # fewer cells for test
    m = 2
    R_inf = 1e-6

    # Create PMLBoundary for "X-"
    pml = PMLBoundary("X-", dt=dt, dx=dx, N_pml=N_pml, m=m, R_inf=R_inf)
    pml.allocate_tensors(Nz, Ny, Nx, device="cpu")

    # Create dummy full‐domain fields
    Hx = torch.ones((Nz + 1, Ny, Nx))
    Hy = torch.ones((Nz, Ny + 1, Nx))
    Hz = torch.ones((Nz, Ny, Nx + 1))
    Ey = torch.rand((Nz + 1, Ny, Nx + 1))
    Ez = torch.rand((Nz + 1, Ny + 1, Nx))
    mu_map = torch.ones((Nz, Ny, Nx)) * 4e-7 * math.pi  # uniform μ

    # Copy Hx before calling
    Hx_before = Hx.clone()

    # Apply PML update on H
    pml.apply_on_H(Hx, Hy, Hz, Ey, Ez, mu_map, dx, dy, dz)

    # Check that in PML region (i=0..N_pml-1) Hx changed (not all ones)
    changed_region = Hx[:, :, :N_pml]
    assert not torch.allclose(changed_region, torch.ones_like(changed_region))

    # Check that outside PML (i >= N_pml) Hx stays as before
    unchanged_region = Hx[:, :, N_pml:]
    assert torch.allclose(unchanged_region, Hx_before[:, :, N_pml:])


def test_pml_boundary_simple_E_update():
    """
    Perform a very basic smoke test of apply_on_E for the PMLBoundary on face X-.
    We initialize Ex with ones in the PML region and Hx, Hy with arbitrary patterns,
    then call apply_on_E and check that Ex is no longer all ones in the PML. This
    verifies the method executes without error and modifies Ex in the intended region.
    """
    Nz, Ny, Nx = 6, 6, 6
    dt = 1e-12
    dx = dy = dz = 1e-3
    N_pml = 2
    m = 2
    R_inf = 1e-6

    # Create PMLBoundary for "X-"
    pml = PMLBoundary("X-", dt=dt, dx=dx, N_pml=N_pml, m=m, R_inf=R_inf)
    pml.allocate_tensors(Nz, Ny, Nx, device="cpu")

    # Create dummy full‐domain fields
    Ex = torch.ones((Nz, Ny + 1, Nx + 1))
    Ey = torch.rand((Nz + 1, Ny, Nx + 1))
    Ez = torch.rand((Nz + 1, Ny + 1, Nx))
    Hx = torch.rand((Nz + 1, Ny, Nx))
    Hy = torch.rand((Nz, Ny + 1, Nx))
    Hz = torch.rand((Nz, Ny, Nx + 1))
    eps_map = torch.ones((Nz, Ny, Nx)) * 8.854e-12  # uniform ε

    # Copy Ex before calling
    Ex_before = Ex.clone()

    # Apply PML update on E
    pml.apply_on_E(Ex, Ey, Ez, Hx, Hy, Hz, eps_map, dx, dy, dz)

    # Check that in PML region (i=0..N_pml-1) Ex changed (not all ones)
    changed_region = Ex[:, :, :N_pml]
    assert not torch.allclose(changed_region, torch.ones_like(changed_region))

    # Check that outside PML (i >= N_pml) Ex remains as before
    unchanged_region = Ex[:, :, N_pml:]
    assert torch.allclose(unchanged_region, Ex_before[:, :, N_pml:])


if __name__ == "__main__":
    pytest.main()
