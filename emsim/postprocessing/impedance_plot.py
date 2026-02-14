"""Plotting functions for input impedance analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Optional


def plot_impedance(impedance_data: Union[str, Path, dict], 
                   save_path: Optional[Union[str, Path]] = None):
    """Plot input impedance: Re(Z), Im(Z), |Z|, and phase.
    
    Parameters
    ----------
    impedance_data : str, Path, or dict
        Either path to CSV file with columns (freq_Hz, Z_real, Z_imag)
        or dictionary with keys ('freqs', 'Z_in').
    save_path : str or Path, optional
        Path to save the figure. If None, displays the plot.
    
    Examples
    --------
    From CSV file:
    >>> plot_impedance("outputs/impedance.csv", "outputs/impedance.png")
    
    From simulation result:
    >>> result = solver.run()
    >>> plot_impedance({'freqs': result['freqs'], 'Z_in': result['Z_in']})
    """
    # Load data
    if isinstance(impedance_data, dict):
        freqs = impedance_data['freqs']
        Z_in = impedance_data['Z_in']
    else:
        df = pd.read_csv(impedance_data)
        freqs = df['freq_Hz'].values
        Z_in = df['Z_real'].values + 1j * df['Z_imag'].values
    
    # Compute impedance components
    Z_real = np.real(Z_in)
    Z_imag = np.imag(Z_in)
    Z_mag = np.abs(Z_in)
    Z_phase = np.angle(Z_in, deg=True)
    
    freq_GHz = freqs / 1e9
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    # Real part
    axes[0, 0].plot(freq_GHz, Z_real, 'b-', linewidth=2)
    axes[0, 0].axhline(50, color='r', linestyle='--', alpha=0.5, label='50 Ω')
    axes[0, 0].set_ylabel('Re(Z) [Ω]', fontsize=11)
    axes[0, 0].set_title('Real Part (Resistance)', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Imaginary part
    axes[0, 1].plot(freq_GHz, Z_imag, 'r-', linewidth=2)
    axes[0, 1].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[0, 1].set_ylabel('Im(Z) [Ω]', fontsize=11)
    axes[0, 1].set_title('Imaginary Part (Reactance)', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Magnitude
    axes[1, 0].plot(freq_GHz, Z_mag, 'g-', linewidth=2)
    axes[1, 0].axhline(50, color='r', linestyle='--', alpha=0.5, label='50 Ω')
    axes[1, 0].set_xlabel('Frequency [GHz]', fontsize=11)
    axes[1, 0].set_ylabel('|Z| [Ω]', fontsize=11)
    axes[1, 0].set_title('Magnitude', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Phase
    axes[1, 1].plot(freq_GHz, Z_phase, 'purple', linewidth=2)
    axes[1, 1].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Frequency [GHz]', fontsize=11)
    axes[1, 1].set_ylabel('Phase [deg]', fontsize=11)
    axes[1, 1].set_title('Phase', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle('Input Impedance Z_in', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_smith_chart(impedance_data: Union[str, Path, dict],
                     save_path: Optional[Union[str, Path]] = None,
                     Z0: float = 50.0):
    """Plot impedance on a Smith chart.
    
    Parameters
    ----------
    impedance_data : str, Path, or dict
        Impedance data (same format as plot_impedance).
    save_path : str or Path, optional
        Path to save the figure.
    Z0 : float
        Characteristic impedance for normalization (default 50 Ω).
    """
    # Load data
    if isinstance(impedance_data, dict):
        freqs = impedance_data['freqs']
        Z_in = impedance_data['Z_in']
    else:
        df = pd.read_csv(impedance_data)
        freqs = df['freq_Hz'].values
        Z_in = df['Z_real'].values + 1j * df['Z_imag'].values
    
    # Normalize impedance
    z_norm = Z_in / Z0
    
    # Reflection coefficient
    gamma = (z_norm - 1) / (z_norm + 1)
    
    # Smith chart plot (simplified - full implementation would use smithplot library)
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot trajectory
    theta = np.angle(gamma)
    r = np.abs(gamma)
    ax.plot(theta, r, 'b-', linewidth=2, label='Impedance locus')
    ax.scatter(theta[0], r[0], c='g', s=100, marker='o', label=f'{freqs[0]/1e9:.2f} GHz')
    ax.scatter(theta[-1], r[-1], c='r', s=100, marker='s', label=f'{freqs[-1]/1e9:.2f} GHz')
    
    ax.set_ylim(0, 1)
    ax.set_title('Smith Chart (Simplified)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right')
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
