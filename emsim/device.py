"""Device selection (GPU/CPU) and performance options for TensorFlow-backed FDTD.

Call setup_device() once at the start of a simulation (e.g. in
Simulation.build()) so that all subsequent tensors are created on the
chosen device. Optional XLA and mixed-precision settings can improve
throughput on GPU.
"""

import logging
from typing import Literal

import tensorflow as tf


def setup_device(
    choice: Literal["auto", "cpu", "gpu"] = "auto",
    enable_xla: bool = False,
    mixed_precision: bool = False,
) -> str:
    """Configure TensorFlow device and optional performance options.

    Parameters
    ----------
    choice : {'auto', 'cpu', 'gpu'}
        - **auto**: use GPU if available, otherwise CPU.
        - **cpu**: force CPU (hide GPU from TensorFlow).
        - **gpu**: use GPU if available; if none, log warning and use CPU.
    enable_xla : bool, optional
        Enable XLA JIT compilation for faster execution. Default False.
    mixed_precision : bool, optional
        Use mixed_float16 policy on GPU for higher throughput. Default False.
        Ignored on CPU.

    Returns
    -------
    str
        "CPU" or "GPU".

    Notes
    -----
    Call this once before creating the grid/solver. XLA and mixed precision
    can yield 2–5x speedup on GPU but may change numerical results slightly.
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
        device = "GPU"
    else:
        device = "GPU" if gpus else "CPU"

    if enable_xla:
        tf.config.optimizer.set_jit(True)

    if mixed_precision and gpus:
        try:
            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(policy)
        except Exception as e:
            logger.warning("Mixed precision not set: %s", e)

    return device
