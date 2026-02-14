"""Diagnostic: plot time-domain signals to understand S-parameter issue.

Load the saved result and plot incident, reflected, and transmitted time series
to verify what we're recording at each port.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

def main():
    # Load result metadata
    meta = pd.read_csv(OUTPUT_DIR / "result_metadata.csv")
    n_steps_run = int(meta["n_steps_run"].iloc[0])
    
    # Load S-parameters CSV
    sp = pd.read_csv(OUTPUT_DIR / "s_parameters.csv")
    
    print("=== S-parameters diagnostic ===")
    print(f"n_steps_run: {n_steps_run}")
    print(f"Frequency points: {len(sp)}")
    print(f"\nS11 range: {sp['S11_dB'].min():.2f} to {sp['S11_dB'].max():.2f} dB")
    print(f"S21 range: {sp['S21_dB'].min():.2f} to {sp['S21_dB'].max():.2f} dB")
    print(f"\nExpected for lossless uniform waveguide:")
    print(f"  S11: < -20 dB (low reflection)")
    print(f"  S21: ~ 0 dB (full transmission)")
    print(f"\nCurrent result suggests:")
    if sp['S11_dB'].mean() > -10:
        print(f"  - S11 too high")
    if sp['S21_dB'].mean() < -10:
        print(f"  - S21 too low")
    
    print("\n=== Next steps ===")
    print("1. Add time-series output to simulation")
    print("2. Verify mode_norm_sq calculation")
    print("3. Check port positions")
    print("4. Consider reference-simulation method")

if __name__ == "__main__":
    main()
