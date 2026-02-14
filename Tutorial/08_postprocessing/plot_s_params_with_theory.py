#!/usr/bin/env python3
"""
Tutorial 08 — S-parameters com teoria (fc TE10).

Carrega s_parameters.csv de um output de simulação (ex.: WR42), plota |S11| e |S21| em dB
e sobrepõe a frequência de corte TE10 (teoria) para guia rectangular.

Uso (a partir da raiz do projeto):
  python Tutorial/08_postprocessing/plot_s_params_with_theory.py [output_dir]

Se output_dir for omitido, usa Simulations/WR42/outputs (corra a simulação antes).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Tutorial.common.theory import te10_cutoff

# WR42 width [m]
A_WR42 = 10.67e-3


def load_s_params(output_dir: Path) -> tuple:
    """Load frequency and S11, S21 from s_parameters.csv. Returns (freqs, S11, S21)."""
    sp_path = output_dir / "s_parameters.csv"
    if not sp_path.is_file():
        raise FileNotFoundError(f"Not found: {sp_path}. Run the simulation first.")
    sp = pd.read_csv(sp_path)
    freqs = sp["frequency_Hz"].values.astype(np.float64)
    S11 = sp["S11_real"].values + 1j * sp["S11_imag"].values
    S21 = sp["S21_real"].values + 1j * sp["S21_imag"].values
    return freqs, S11, S21


def main():
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1]).resolve()
    else:
        output_dir = ROOT / "Simulations" / "WR42" / "outputs"

    if not output_dir.is_dir():
        print(f"Output directory not found: {output_dir}")
        print("Run: python Simulations/WR42/run.py  (then run this script again)")
        sys.exit(1)

    freqs, S11, S21 = load_s_params(output_dir)
    fc = te10_cutoff(A_WR42)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(freqs / 1e9, 20 * np.log10(np.abs(S11) + 1e-12), label="|S11| (dB)")
    ax.plot(freqs / 1e9, 20 * np.log10(np.abs(S21) + 1e-12), label="|S21| (dB)")
    ax.axvline(fc / 1e9, color="red", linestyle="--", label=f"fc TE10 = {fc/1e9:.2f} GHz (teoria)")
    ax.set_xlabel("Frequencia [GHz]")
    ax.set_ylabel("dB")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("S-parameters: simulacao vs frequencia de corte TE10 (guia WR42)")
    plt.tight_layout()
    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / "s_params_with_fc_te10.png", dpi=120)
    plt.close()
    print(f"Guardado: {out_dir / 's_params_with_fc_te10.png'}")


if __name__ == "__main__":
    main()
