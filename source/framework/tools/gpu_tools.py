"""
Purpose
-------
- Get GPU information.
- Enforce particular GPU for Tensorflow.
- Limit GPU memory usage.

Contributors
------------
- TMS-Namespace
"""

from typing import List, Optional

import tensorflow as tf
import nvidia_smi


def get_gpu_info(
    gpu_index: int,
) -> tuple[int, int, str, float]:
    """
    Get GPU information of a specific GPU from Nvidia SMI.

    Arguments
    ---------
    - `gpu_index`: Index of the GPU to get information about.

    Returns
    --------
    - The number of GPUs on the system.
    - The requested index of the GPU to get information about.
    - The name of the GPU.
    - The total available memory of the GPU in Gb.

    Raises
    ------
    - Exception: If no GPUs are found.
    """
    try:
        nvidia_smi.nvmlInit()

        device_count = nvidia_smi.nvmlDeviceGetCount()

        if device_count > 0:
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_index)
            gpu_name = nvidia_smi.nvmlDeviceGetName(handle)
            total_memory_gb = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).total / (
                1024**3
            )

            return (device_count, gpu_index, gpu_name, total_memory_gb)
        else:
            raise Exception("No GPUs found.")

    except Exception as e:
        raise e
    finally:
        # Shutdown the NVIDIA System Management Interface
        nvidia_smi.nvmlShutdown()


def config_gpu(
    gpu_index: Optional[float] = None, memory_limit_gb: Optional[float] = None
) -> Optional[List[str]]:
    """
    Select the GPU that TensorFlow, and how much memory it will reserve.

    Arguments
    ---------
    - `gpu_index`: the index of the GPU that we want to use, if smaller than
    zero, it will do nothing.
    - `memory_limit`: the memory limit of the GPU that we want to use in Gb.

    Notes
    -----
    - If `gpu_index` is None, then Tensorflow will select a GPU automatically.
    - If `memory_limit` is None, then whole GPU memory will be allocated.

    Returns
    -------
    - None or List of available GPUs.

    Raises
    ------
    - Exception: If no GPUs are found.
    """

    gpus = tf.config.list_physical_devices("GPU")

    if len(gpus) == 0:
        raise Exception("No GPUs found.")

    if gpu_index is not None:
        tf.config.experimental.set_visible_devices(
            devices=gpus[gpu_index], device_type="GPU"
        )

    gpu_index = 0 if gpu_index is None else gpu_index

    if memory_limit_gb is not None:
        for gpu in gpus:
            tf.config.set_logical_device_configuration(
                device=gpu,
                logical_devices=[
                    tf.config.LogicalDeviceConfiguration(
                        memory_limit=1024 * memory_limit_gb
                    )
                ],
            )

    return gpus
