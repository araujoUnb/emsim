"""Near-field to far-field (NF2FF) transformation for antenna analysis.

This module implements the equivalence principle to compute far-field radiation
patterns from near-field data recorded on a closed surface around the antenna.
"""

from typing import Dict, Any
import numpy as np


class NF2FFBox:
    """Recording box for near-field to far-field transformation.
    
    Records tangential E and H field components on the six faces of a
    rectangular box surrounding the antenna structure.
    
    Parameters
    ----------
    i_range : tuple[int, int]
        Grid index range in x-direction (i_min, i_max).
    j_range : tuple[int, int]
        Grid index range in y-direction (j_min, j_max).
    k_range : tuple[int, int]
        Grid index range in z-direction (k_min, k_max).
    
    Attributes
    ----------
    faces : dict
        Dictionary storing field data for each face ('x-', 'x+', 'y-', 'y+', 'z-', 'z+').
        Each face has 'E_tan' and 'H_tan' lists of temporal snapshots.
    """
    
    def __init__(self, i_range: tuple, j_range: tuple, k_range: tuple):
        self.i_range = i_range
        self.j_range = j_range
        self.k_range = k_range
        
        # Initialize storage for each face
        self.faces = {
            'x-': {'E_tan': [], 'H_tan': []},
            'x+': {'E_tan': [], 'H_tan': []},
            'y-': {'E_tan': [], 'H_tan': []},
            'y+': {'E_tan': [], 'H_tan': []},
            'z-': {'E_tan': [], 'H_tan': []},
            'z+': {'E_tan': [], 'H_tan': []},
        }
    
    def record(self, Ex, Ey, Ez, Hx, Hy, Hz):
        """Record tangential field components on all six faces.
        
        Parameters
        ----------
        Ex, Ey, Ez : tf.Variable, shape [Nz, Ny, Nx]
            Electric field components.
        Hx, Hy, Hz : tf.Variable, shape [Nz, Ny, Nx]
            Magnetic field components.
        
        Notes
        -----
        For each face, records the two tangential components:
        - x-faces (normal in x): tangential are Ey, Ez, Hy, Hz
        - y-faces (normal in y): tangential are Ex, Ez, Hx, Hz
        - z-faces (normal in z): tangential are Ex, Ey, Hx, Hy
        """
        i_min, i_max = self.i_range
        j_min, j_max = self.j_range
        k_min, k_max = self.k_range
        
        # Face x- (i = i_min, normal in -x direction)
        self.faces['x-']['E_tan'].append({
            'Ey': Ey[k_min:k_max, j_min:j_max, i_min].numpy().copy(),
            'Ez': Ez[k_min:k_max, j_min:j_max, i_min].numpy().copy(),
        })
        self.faces['x-']['H_tan'].append({
            'Hy': Hy[k_min:k_max, j_min:j_max, i_min].numpy().copy(),
            'Hz': Hz[k_min:k_max, j_min:j_max, i_min].numpy().copy(),
        })
        
        # Face x+ (i = i_max, normal in +x direction)
        self.faces['x+']['E_tan'].append({
            'Ey': Ey[k_min:k_max, j_min:j_max, i_max-1].numpy().copy(),
            'Ez': Ez[k_min:k_max, j_min:j_max, i_max-1].numpy().copy(),
        })
        self.faces['x+']['H_tan'].append({
            'Hy': Hy[k_min:k_max, j_min:j_max, i_max-1].numpy().copy(),
            'Hz': Hz[k_min:k_max, j_min:j_max, i_max-1].numpy().copy(),
        })
        
        # Face y- (j = j_min, normal in -y direction)
        self.faces['y-']['E_tan'].append({
            'Ex': Ex[k_min:k_max, j_min, i_min:i_max].numpy().copy(),
            'Ez': Ez[k_min:k_max, j_min, i_min:i_max].numpy().copy(),
        })
        self.faces['y-']['H_tan'].append({
            'Hx': Hx[k_min:k_max, j_min, i_min:i_max].numpy().copy(),
            'Hz': Hz[k_min:k_max, j_min, i_min:i_max].numpy().copy(),
        })
        
        # Face y+ (j = j_max, normal in +y direction)
        self.faces['y+']['E_tan'].append({
            'Ex': Ex[k_min:k_max, j_max-1, i_min:i_max].numpy().copy(),
            'Ez': Ez[k_min:k_max, j_max-1, i_min:i_max].numpy().copy(),
        })
        self.faces['y+']['H_tan'].append({
            'Hx': Hx[k_min:k_max, j_max-1, i_min:i_max].numpy().copy(),
            'Hz': Hz[k_min:k_max, j_max-1, i_min:i_max].numpy().copy(),
        })
        
        # Face z- (k = k_min, normal in -z direction)
        self.faces['z-']['E_tan'].append({
            'Ex': Ex[k_min, j_min:j_max, i_min:i_max].numpy().copy(),
            'Ey': Ey[k_min, j_min:j_max, i_min:i_max].numpy().copy(),
        })
        self.faces['z-']['H_tan'].append({
            'Hx': Hx[k_min, j_min:j_max, i_min:i_max].numpy().copy(),
            'Hy': Hy[k_min, j_min:j_max, i_min:i_max].numpy().copy(),
        })
        
        # Face z+ (k = k_max, normal in +z direction)
        self.faces['z+']['E_tan'].append({
            'Ex': Ex[k_max-1, j_min:j_max, i_min:i_max].numpy().copy(),
            'Ey': Ey[k_max-1, j_min:j_max, i_min:i_max].numpy().copy(),
        })
        self.faces['z+']['H_tan'].append({
            'Hx': Hx[k_max-1, j_min:j_max, i_min:i_max].numpy().copy(),
            'Hy': Hy[k_max-1, j_min:j_max, i_min:i_max].numpy().copy(),
        })
    
    def reset(self):
        """Clear all recorded field data."""
        for face_data in self.faces.values():
            face_data['E_tan'].clear()
            face_data['H_tan'].clear()


def compute_nf2ff(nf2ff_box: NF2FFBox, freq: float,
                  theta: np.ndarray, phi: np.ndarray,
                  grid_info: Dict[str, Any]) -> Dict[str, Any]:
    """Compute far-field radiation pattern using equivalence principle.
    
    Transforms near-field data recorded on a closed box to far-field
    radiation patterns via Huygens' equivalence principle and Green's functions.
    
    Parameters
    ----------
    nf2ff_box : NF2FFBox
        Box containing recorded near-field data.
    freq : float
        Frequency for far-field calculation [Hz].
    theta : np.ndarray
        Elevation angles [degrees], typically np.arange(-180, 180, 2).
    phi : np.ndarray
        Azimuth angles [degrees], typically [0, 90] for E-plane and H-plane cuts.
    grid_info : dict
        Grid information containing:
        - 'dx', 'dy', 'dz': grid spacings [m]
        - 'dt': time step [s]
        - Other grid parameters as needed
    
    Returns
    -------
    dict
        Far-field results containing:
        - 'E_theta': complex array [len(theta), len(phi)] - theta component
        - 'E_phi': complex array [len(theta), len(phi)] - phi component
        - 'E_norm': float array - total field magnitude |E|
        - 'directivity': float array [len(theta), len(phi)] - directivity [dBi]
        - 'Dmax': float - maximum directivity [dBi]
        - 'theta': input theta array
        - 'phi': input phi array
        - 'freq': input frequency
    
    Notes
    -----
    This is a simplified implementation. A complete implementation would:
    1. FFT temporal data to frequency domain
    2. Compute equivalent currents: J_eq = n × H, M_eq = -n × E
    3. Apply spherical Green's function for far-field
    4. Integrate over all six faces
    5. Compute directivity from radiated power
    
    For now, returns placeholder data structure. Full implementation is complex
    and would require ~200 lines of numerical integration code.
    """
    # TODO: Implement full NF2FF transformation
    # This is a placeholder that returns the correct structure
    
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)
    
    # Create meshgrid for 2D patterns
    THETA, PHI = np.meshgrid(theta_rad, phi_rad, indexing='ij')
    
    # Placeholder: return zeros with correct shape
    # Real implementation would compute E-fields from surface integrals
    E_theta = np.zeros((len(theta), len(phi)), dtype=complex)
    E_phi = np.zeros((len(theta), len(phi)), dtype=complex)
    
    # Placeholder directivity (would be computed from Poynting vector)
    E_norm = np.abs(E_theta) + np.abs(E_phi)
    directivity_linear = E_norm / np.max(E_norm) if np.max(E_norm) > 0 else E_norm
    directivity_dBi = 10 * np.log10(directivity_linear + 1e-12)
    Dmax = float(np.max(directivity_dBi))
    
    return {
        'E_theta': E_theta,
        'E_phi': E_phi,
        'E_norm': E_norm,
        'directivity': directivity_dBi,
        'Dmax': Dmax,
        'theta': theta,
        'phi': phi,
        'freq': freq,
        'note': 'NF2FF transformation not yet fully implemented. This is a placeholder.',
    }
