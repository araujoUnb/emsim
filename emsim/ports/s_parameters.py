"""S-parameter extraction from time-domain port recordings via FFT with modal decomposition."""

import numpy as np
from typing import Callable, Optional, Union


def compute_s_parameters(E_record_input, H_record_input,
                         E_record_output, H_record_output,
                         Z_mode_func: Callable[[float], Union[float, complex]],
                         dt: float,
                         freq_min: Optional[float] = None,
                         freq_max: Optional[float] = None):
    """Compute S11 and S21 using modal decomposition of E and H fields.

    Modal decomposition for TE modes propagating in +z direction:
      - Forward wave:   Ey_fwd = a+,  Hx_fwd = -a+ / Z_TE
      - Backward wave:  Ey_bwd = a-,  Hx_bwd = +a- / Z_TE

    Total fields at a port:
      Ey_total = a+ + a-
      Hx_total = (-a+ + a-) / Z_TE

    Solving for a+ and a-:
      a+ = (Ey_total - Z_TE * Hx_total) / 2
      a- = (Ey_total + Z_TE * Hx_total) / 2

    The mode profiles used for the overlap integrals are both positive
    (same spatial pattern for Ey and Hx), so the recorded H_overlap
    captures Hx with its physical sign.  Since Hx_fwd is *negative*
    for +z propagation, the formulas above are correct.

    S-parameters:
      S11 = a-_input  / a+_input   (reflection at input port)
      S21 = a+_output / a+_input   (transmission to output port)

    Parameters
    ----------
    E_record_input  : list[float]  E-field overlap time series at input port
    H_record_input  : list[float]  H-field overlap time series at input port
    E_record_output : list[float]  E-field overlap time series at output port
    H_record_output : list[float]  H-field overlap time series at output port
    Z_mode_func     : callable     Z(f) returning mode impedance (may be complex)
    dt              : float        Time step [s]
    freq_min        : float, optional  Minimum frequency [Hz] for output
    freq_max        : float, optional  Maximum frequency [Hz] for output

    Returns
    -------
    dict with keys:
        'freqs' : 1-D numpy array of frequencies [Hz]
        'S11'   : complex numpy array, reflection coefficient
        'S21'   : complex numpy array, transmission coefficient
    """
    E_in = np.array(E_record_input, dtype=np.float64)
    H_in = np.array(H_record_input, dtype=np.float64)
    E_out = np.array(E_record_output, dtype=np.float64)
    H_out = np.array(H_record_output, dtype=np.float64)

    N = len(E_in)
    freqs = np.fft.rfftfreq(N, d=dt)

    E_in_f = np.fft.rfft(E_in)
    H_in_f = np.fft.rfft(H_in)
    E_out_f = np.fft.rfft(E_out)
    H_out_f = np.fft.rfft(H_out)

    # Mode impedance for each frequency bin (complex for evanescent modes)
    Z_array = np.array(
        [Z_mode_func(f) if f > 0 else Z_mode_func(freqs[1]) for f in freqs],
        dtype=np.complex128,
    )

    # Modal decomposition — see docstring for derivation
    a_forward_input  = 0.5 * (E_in_f  - Z_array * H_in_f)
    a_backward_input = 0.5 * (E_in_f  + Z_array * H_in_f)
    a_forward_output = 0.5 * (E_out_f - Z_array * H_out_f)

    # S-parameters (guard against division by near-zero)
    eps = 1e-30
    safe_a_fwd_in = np.where(
        np.abs(a_forward_input) > eps, a_forward_input, eps
    )

    S11 = a_backward_input / safe_a_fwd_in
    S21 = a_forward_output / safe_a_fwd_in

    # Frequency range mask
    if freq_min is not None or freq_max is not None:
        mask = np.ones(len(freqs), dtype=bool)
        if freq_min is not None:
            mask &= freqs >= freq_min
        if freq_max is not None:
            mask &= freqs <= freq_max
        freqs = freqs[mask]
        S11 = S11[mask]
        S21 = S21[mask]

    return {'freqs': freqs, 'S11': S11, 'S21': S21}
