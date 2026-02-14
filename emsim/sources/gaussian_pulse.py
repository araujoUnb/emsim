"""Gaussian-modulated sinusoidal pulse source for FDTD excitation."""

import math

import tensorflow as tf


class GaussianPulse:
    """Broadband Gaussian pulse with centred carrier.

    s(t) = exp(-(t-t0)^2 / (2*tau^2)) * sin(w0*(t-t0)).

    The carrier is centred under the envelope so the pulse peak coincides
    with the carrier peak.  Using sin ensures the waveform starts from zero
    at t=0 (the Gaussian envelope is negligible there).

    Parameters
    ----------
    f0 : float
        Centre frequency [Hz].
    bandwidth : float
        Bandwidth [Hz] (defines the Gaussian envelope width).
    """

    def __init__(self, f0: float, bandwidth: float):
        self.f0 = f0
        self.bandwidth = bandwidth
        self.omega0 = 2.0 * math.pi * f0
        self.tau = 1.0 / (2.0 * math.pi * bandwidth)
        self.t0 = 4.5 * self.tau  # delay so pulse is ~0 at t=0

    def __call__(self, t):
        """Evaluate the pulse at time(s) t.

        Parameters
        ----------
        t : tf.Tensor or float
            Time value(s) in seconds.

        Returns
        -------
        tf.Tensor  (same shape as t)
        """
        t = tf.cast(t, tf.float32)
        t0 = tf.constant(self.t0, dtype=tf.float32)
        tau = tf.constant(self.tau, dtype=tf.float32)
        omega0 = tf.constant(self.omega0, dtype=tf.float32)
        envelope = tf.exp(-((t - t0) ** 2) / (2.0 * tau ** 2))
        carrier = tf.sin(omega0 * (t - t0))
        return envelope * carrier

    @property
    def duration(self):
        """Recommended simulation duration to capture the full pulse [s]."""
        return 2.0 * self.t0 + 5.0 * self.tau
