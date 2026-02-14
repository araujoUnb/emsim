"""Device selection (GPU/CPU) for TensorFlow-backed FDTD.

Call setup_device() once at the start of a simulation (e.g. in
Simulation.build()) so that all subsequent tensors are created on the
chosen device.
"""

import logging
from typing import Literal

import tensorflow as tf


def setup_device(choice: Literal["auto", "cpu", "gpu"] = "auto") -> str:
    """Configure TensorFlow device and return the one in use.

    Parameters
    ----------
    choice : {'auto', 'cpu', 'gpu'}
        - **auto**: use GPU if available, otherwise CPU.
        - **cpu**: force CPU (hide GPU from TensorFlow).
        - **gpu**: use GPU if available; if none, log warning and use CPU.

    Returns
    -------
    str
        "CPU" or "GPU".

    Notes
    -----
    Call this once before creating the grid/solver. In environments without
    GPU or with incorrect CUDA drivers, use device: cpu to avoid failures.
    """
    logger = logging.getLogger(__name__)
    gpus = tf.config.list_physical_devices("GPU")

    if choice == "cpu":
        tf.config.set_visible_devices([], "GPU")
        return "CPU"

    if choice == "gpu":
        if not gpus:
            logger.warning(
                "device 'gpu' requested but no GPU found; falling back to CPU."
            )
            return "CPU"
        return "GPU"

    # auto
    if gpus:
        return "GPU"
    return "CPU"
