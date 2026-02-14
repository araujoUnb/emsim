"""Post-processing for patch antenna simulation.

Generates plots from saved simulation results:
- S11 reflection coefficient
- Input impedance (real, imaginary, magnitude, phase)
- Radiation pattern (if nf2ff was computed)
- Field snapshots

Usage
-----
Run after completing the simulation:
    python Simulations/Patch_Antenna/postprocess.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    output_dir = Path(__file__).resolve().parent / "outputs"
    
    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        print("Run the simulation first with run.py")
        return
    
    print("Generating plots...")
    
    # 1. S11 (reflection coefficient)
    s_params_file = output_dir / "s_parameters.csv"
    if s_params_file.exists():
        plot_s11(s_params_file, save_path=output_dir / "s11.png")
        print(f"  ✓ S11 plot saved: {output_dir / 's11.png'}")
    else:
        print(f"  ✗ S-parameters file not found: {s_params_file}")
    
    # 2. Input impedance (not yet implemented in result saving)
    # TODO: Add impedance data to simulation results
    print("  ⚠ Impedance plot: not yet implemented")
    
    # 3. Radiation pattern (not yet implemented)
    print("  ⚠ Radiation pattern: nf2ff transformation not yet complete")
    
    # 4. Field snapshots
    ez_file = output_dir / "ez_snapshots.csv"
    if ez_file.exists():
        plot_field_snapshots_simple(ez_file, save_path=output_dir / "fields.png")
        print(f"  ✓ Field snapshots saved: {output_dir / 'fields.png'}")
    else:
        print(f"  ⚠ Ez snapshots file not found: {ez_file}")
    
    print("\nAll available plots generated in:", output_dir)


def plot_s11(csv_path: Path, save_path: Path):
    """Plot S11 magnitude and phase."""
    df = pd.read_csv(csv_path)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Magnitude
    ax1.plot(df['frequency_Hz'] / 1e9, df['S11_dB'], 'b-', linewidth=2)
    ax1.axhline(-10, color='r', linestyle='--', alpha=0.5, label='-10 dB')
    ax1.set_ylabel('|S11| [dB]', fontsize=12)
    ax1.set_title('Reflection Coefficient S11', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Phase
    ax2.plot(df['frequency_Hz'] / 1e9, df['S11_phase_deg'], 'g-', linewidth=2)
    ax2.set_xlabel('Frequency [GHz]', fontsize=12)
    ax2.set_ylabel('Phase [deg]', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_field_snapshots_simple(csv_path: Path, save_path: Path, max_snapshots=6):
    """Plot Ez field snapshots in a grid."""
    df = pd.read_csv(csv_path)
    
    # Get unique snapshot indices
    snapshots = df['snapshot_index'].unique()
    n_snapshots = min(len(snapshots), max_snapshots)
    
    if n_snapshots == 0:
        print("No snapshots to plot")
        return
    
    # Create subplot grid
    cols = 3
    rows = (n_snapshots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    axes = np.atleast_2d(axes).ravel()
    
    for idx, snap_idx in enumerate(snapshots[:n_snapshots]):
        snap_data = df[df['snapshot_index'] == snap_idx]
        
        # Reshape to 2D (assuming i is x, k is z)
        i_vals = snap_data['i'].values
        k_vals = snap_data['k'].values
        Ez_vals = snap_data['Ez'].values
        
        ni = len(np.unique(i_vals))
        nk = len(np.unique(k_vals))
        
        Ez_2d = Ez_vals.reshape(nk, ni)
        
        # Plot
        im = axes[idx].imshow(Ez_2d, aspect='auto', cmap='RdBu', origin='lower')
        axes[idx].set_title(f'Snapshot {snap_idx}', fontsize=10)
        axes[idx].set_xlabel('x')
        axes[idx].set_ylabel('z')
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for idx in range(n_snapshots, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Ez Field Evolution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
