"""Plotting functions for far-field radiation patterns."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Optional


def plot_radiation_pattern(nf2ff_data: Union[str, Path, dict],
                           save_path: Optional[Union[str, Path]] = None,
                           planes: list = [0, 90]):
    """Plot far-field radiation patterns in polar coordinates.
    
    Parameters
    ----------
    nf2ff_data : str, Path, or dict
        Either path to CSV file or dictionary with nf2ff results.
        Expected keys: 'theta', 'phi', 'directivity', 'E_norm'.
    save_path : str or Path, optional
        Path to save the figure.
    planes : list
        Phi angles [degrees] to plot (default [0, 90] for E-plane and H-plane).
    
    Examples
    --------
    >>> plot_radiation_pattern("outputs/nf2ff_result.csv", 
    ...                        "outputs/radiation_pattern.png")
    """
    # Load data
    if isinstance(nf2ff_data, dict):
        theta = nf2ff_data['theta']
        phi_vals = nf2ff_data['phi']
        directivity = nf2ff_data['directivity']
    else:
        df = pd.read_csv(nf2ff_data)
        # Assumes columns: theta_deg, phi_deg, directivity_dBi
        theta = df['theta_deg'].unique()
        phi_vals = df['phi_deg'].unique()
        # Reshape directivity to 2D
        directivity = df['directivity_dBi'].values.reshape(len(theta), len(phi_vals))
    
    # Create polar subplots
    n_planes = len(planes)
    fig = plt.figure(figsize=(6 * n_planes, 6))
    
    for idx, phi_cut in enumerate(planes):
        ax = fig.add_subplot(1, n_planes, idx + 1, projection='polar')
        
        # Find closest phi index
        if isinstance(nf2ff_data, dict):
            phi_idx = np.argmin(np.abs(phi_vals - phi_cut))
        else:
            phi_idx = np.where(phi_vals == phi_cut)[0][0]
        
        # Extract pattern for this phi cut
        pattern = directivity[:, phi_idx]
        theta_rad = np.deg2rad(theta)
        
        # Normalize to 0 dB max
        pattern_norm = pattern - np.max(pattern)
        
        # Plot pattern
        ax.plot(theta_rad, pattern_norm, 'b-', linewidth=2)
        ax.fill_between(theta_rad, pattern_norm, -30, alpha=0.3)
        
        # Formatting
        ax.set_theta_zero_location('N')  # 0° at top
        ax.set_theta_direction(-1)  # Clockwise
        ax.set_ylim(-30, 0)
        ax.set_yticks(np.arange(-30, 1, 5))
        ax.set_ylabel('Gain [dBi]', fontsize=10)
        
        # Title based on plane
        if phi_cut == 0:
            plane_name = 'E-plane (xz, φ=0°)'
        elif phi_cut == 90:
            plane_name = 'H-plane (yz, φ=90°)'
        else:
            plane_name = f'φ={phi_cut}° plane'
        
        ax.set_title(plane_name, fontsize=12, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Far-Field Radiation Pattern', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_3d_pattern(nf2ff_data: Union[str, Path, dict],
                    save_path: Optional[Union[str, Path]] = None):
    """Plot 3D radiation pattern.
    
    Parameters
    ----------
    nf2ff_data : str, Path, or dict
        NF2FF results with full theta/phi grid.
    save_path : str or Path, optional
        Path to save the figure.
    """
    # Load data
    if isinstance(nf2ff_data, dict):
        theta = nf2ff_data['theta']
        phi = nf2ff_data['phi']
        E_norm = nf2ff_data['E_norm']
    else:
        df = pd.read_csv(nf2ff_data)
        theta = df['theta_deg'].unique()
        phi = df['phi_deg'].unique()
        E_norm = df['E_norm'].values.reshape(len(theta), len(phi))
    
    # Convert to spherical coordinates
    THETA, PHI = np.meshgrid(np.deg2rad(theta), np.deg2rad(phi), indexing='ij')
    
    # Normalize
    E_norm_normalized = E_norm / np.max(E_norm)
    
    # Convert to Cartesian
    X = E_norm_normalized * np.sin(THETA) * np.cos(PHI)
    Y = E_norm_normalized * np.sin(THETA) * np.sin(PHI)
    Z = E_norm_normalized * np.cos(THETA)
    
    # Plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                          linewidth=0, antialiased=True)
    
    ax.set_xlabel('X', fontsize=11)
    ax.set_ylabel('Y', fontsize=11)
    ax.set_zlabel('Z', fontsize=11)
    ax.set_title('3D Radiation Pattern', fontsize=14, fontweight='bold')
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Normalized E-field')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_directivity_vs_angle(nf2ff_data: Union[str, Path, dict],
                              save_path: Optional[Union[str, Path]] = None):
    """Plot directivity vs angle in rectangular coordinates.
    
    Parameters
    ----------
    nf2ff_data : str, Path, or dict
        NF2FF results.
    save_path : str or Path, optional
        Path to save the figure.
    """
    # Load data
    if isinstance(nf2ff_data, dict):
        theta = nf2ff_data['theta']
        phi_vals = nf2ff_data['phi']
        directivity = nf2ff_data['directivity']
    else:
        df = pd.read_csv(nf2ff_data)
        theta = df['theta_deg'].unique()
        phi_vals = df['phi_deg'].unique()
        directivity = df['directivity_dBi'].values.reshape(len(theta), len(phi_vals))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot for each phi
    for idx, phi_val in enumerate([0, 90]):
        if phi_val in phi_vals or (isinstance(phi_vals, np.ndarray) and phi_val in phi_vals):
            if isinstance(nf2ff_data, dict):
                phi_idx = np.argmin(np.abs(phi_vals - phi_val))
            else:
                phi_idx = np.where(phi_vals == phi_val)[0][0]
            
            pattern = directivity[:, phi_idx]
            label = f'φ={phi_val}° ({"E" if phi_val == 0 else "H"}-plane)'
            ax.plot(theta, pattern, linewidth=2, label=label)
    
    ax.set_xlabel('Theta [degrees]', fontsize=12)
    ax.set_ylabel('Directivity [dBi]', fontsize=12)
    ax.set_title('Directivity vs Elevation Angle', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xlim([-180, 180])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
