"""Plot and export S-parameters (S11, S21) from FDTD simulation result."""

import csv
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_s_parameters_csv(
    result: dict,
    save_path: str,
    f_min: Optional[float] = None,
    f_max: Optional[float] = None,
) -> None:
    """Save S-parameters to a CSV file (frequency, complex S11/S21, magnitudes in dB).

    Parameters
    ----------
    result : dict
        Result dict with keys 'freqs', 'S11', 'S21'.
    save_path : str
        Output path for the CSV file (e.g. outputs/s_parameters.csv).
    f_min : float, optional
        Minimum frequency [Hz] to include. If None, use full band.
    f_max : float, optional
        Maximum frequency [Hz] to include. If None, use full band.
    """
    if f_min is not None:
        f_min = float(f_min)
    if f_max is not None:
        f_max = float(f_max)
    freqs = np.asarray(result["freqs"], dtype=np.float64)
    S11 = np.asarray(result["S11"], dtype=np.complex128)
    S21 = np.asarray(result["S21"], dtype=np.complex128)

    mask = np.ones(len(freqs), dtype=bool)
    if f_min is not None:
        mask &= freqs >= f_min
    if f_max is not None:
        mask &= freqs <= f_max

    freqs_out = freqs[mask]
    S11_out = S11[mask]
    S21_out = S21[mask]

    S11_dB = 20 * np.log10(np.abs(S11_out) + 1e-30)
    S21_dB = 20 * np.log10(np.abs(S21_out) + 1e-30)
    S11_phase_deg = np.degrees(np.angle(S11_out))
    S21_phase_deg = np.degrees(np.angle(S21_out))

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "frequency_Hz",
            "S11_real", "S11_imag",
            "S11_dB", "S11_phase_deg",
            "S21_real", "S21_imag",
            "S21_dB", "S21_phase_deg",
        ])
        for i in range(len(freqs_out)):
            w.writerow([
                f"{freqs_out[i]:.12e}",
                f"{S11_out[i].real:.6e}", f"{S11_out[i].imag:.6e}",
                f"{S11_dB[i]:.6f}", f"{S11_phase_deg[i]:.6f}",
                f"{S21_out[i].real:.6e}", f"{S21_out[i].imag:.6e}",
                f"{S21_dB[i]:.6f}", f"{S21_phase_deg[i]:.6f}",
            ])


def plot_s_parameters(
    result: dict,
    f_min: Optional[float] = None,
    f_max: Optional[float] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot |S11| and |S21| in dB versus frequency.

    Parameters
    ----------
    result : dict
        Result dict from FDTDSolver.run() or Simulation.run(), with keys
        'freqs', 'S11', 'S21' (numpy arrays or compatible).
    f_min : float, optional
        Minimum frequency [Hz] for the plot. If None, use full band.
    f_max : float, optional
        Maximum frequency [Hz] for the plot. If None, use full band.
    save_path : str, optional
        If provided, save the figure to this path (e.g. PNG).
    """
    if f_min is not None:
        f_min = float(f_min)
    if f_max is not None:
        f_max = float(f_max)
    freqs = np.asarray(result["freqs"], dtype=np.float64)
    S11 = np.asarray(result["S11"], dtype=np.complex128)
    S21 = np.asarray(result["S21"], dtype=np.complex128)

    mask = np.ones(len(freqs), dtype=bool)
    if f_min is not None:
        mask &= freqs >= f_min
    if f_max is not None:
        mask &= freqs <= f_max
    freqs_plot = freqs[mask] / 1e9  # GHz
    S11_plot = S11[mask]
    S21_plot = S21[mask]

    S11_dB = 20 * np.log10(np.abs(S11_plot) + 1e-30)
    S21_dB = 20 * np.log10(np.abs(S21_plot) + 1e-30)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(freqs_plot, S11_dB, "b-", linewidth=1.5)
    axes[0].set_xlabel("Frequency [GHz]")
    axes[0].set_ylabel("|S11| [dB]")
    axes[0].set_title("Reflection (S11)")
    axes[0].grid(True)
    if len(S11_dB) > 0:
        s11_min, s11_max = float(np.min(S11_dB)), float(np.max(S11_dB))
        axes[0].set_ylim([min(s11_min - 5, -100), max(s11_max + 2, 0)])
    else:
        axes[0].set_ylim([-60, 0])

    axes[1].plot(freqs_plot, S21_dB, "r-", linewidth=1.5)
    axes[1].set_xlabel("Frequency [GHz]")
    axes[1].set_ylabel("|S21| [dB]")
    axes[1].set_title("Transmission (S21)")
    axes[1].grid(True)
    if len(S21_dB) > 0:
        s21_min, s21_max = float(np.min(S21_dB)), float(np.max(S21_dB))
        axes[1].set_ylim([min(s21_min - 5, -120), max(s21_max + 2, 5)])
    else:
        axes[1].set_ylim([-5, 5])

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
