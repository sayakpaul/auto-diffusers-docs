import subprocess
import psutil
import functools
from torch._inductor.runtime.hints import DeviceProperties
from torch._inductor.utils import get_gpu_type
from typing import Union
import torch


@functools.cache
def get_system_ram_gb():
    """
    Gets the total physical system RAM in Gigabytes.

    Returns:
        float: Total system RAM in GB, or None if it cannot be determined.
    """
    try:
        # Get virtual memory details
        virtual_memory = psutil.virtual_memory()
        # Total physical memory in bytes
        total_ram_bytes = virtual_memory.total
        # Convert bytes to gigabytes (1 GB = 1024^3 bytes)
        total_ram_gb = total_ram_bytes / (1024**3)
        return total_ram_gb
    except Exception as e:
        print(f"Error getting system RAM: {e}")
        return None


@functools.cache
def get_gpu_vram_gb():
    """
    Gets the total GPU VRAM in Gigabytes using the nvidia-smi command.
    This function is intended for NVIDIA GPUs.

    Returns:
        float: Total GPU VRAM in GB, or None if it cannot be determined.
    """
    try:
        # Execute the nvidia-smi command to get GPU memory info
        # The command queries for the total memory and outputs it in MiB
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        # The output will be a string like "12288\n" for the first GPU
        # We take the first line in case there are multiple GPUs
        vram_mib = int(result.stdout.strip().split("\n")[0])
        # Convert MiB to Gigabytes (1 GB = 1024 MiB)
        vram_gb = vram_mib / 1024
        return vram_gb
    except FileNotFoundError:
        # This error occurs if nvidia-smi is not installed or not in the PATH
        print("INFO: 'nvidia-smi' command not found. Cannot determine GPU VRAM.")
        print("      This is expected if you don't have an NVIDIA GPU or drivers installed.")
        return None
    except (subprocess.CalledProcessError, ValueError, IndexError) as e:
        # Handles other potential errors like command failure or parsing issues
        print(f"Error getting GPU VRAM: {e}")
        return None


def categorize_ram(ram_gb):
    """
    Categorizes RAM into 'small', 'medium', or 'large'.

    Args:
        ram_gb (float): The amount of RAM in GB.

    Returns:
        str: The category ('small', 'medium', 'large') or 'unknown'.
    """
    if ram_gb is None:
        return "unknown"
    if ram_gb <= 20:
        return "small"
    elif 20 < ram_gb <= 40:
        return "medium"
    else:  # ram_gb > 40
        return "large"


def categorize_vram(vram_gb):
    """
    Categorizes VRAM into 'small', 'medium', or 'large'.

    Args:
        vram_gb (float): The amount of VRAM in GB.

    Returns:
        str: The category ('small', 'medium', 'large') or 'not applicable/unknown'.
    """
    if vram_gb is None:
        return "not applicable/unknown"
    if vram_gb <= 8:
        return "small"
    elif 8 < vram_gb <= 24:
        return "medium"
    else:  # vram_gb > 24
        return "large"


def main():
    """
    Main function to get and display hardware configuration.
    """
    print("--- System Hardware Configuration Utility ---")

    # Get and categorize System RAM
    ram_gb = get_system_ram_gb()
    ram_category = categorize_ram(ram_gb)
    if ram_gb is not None:
        print(f"\nSystem RAM: {ram_gb:.2f} GB")
        print(f"RAM Category: {ram_category}")
    else:
        print("\nCould not determine System RAM.")

    # Get and categorize GPU VRAM
    vram_gb = get_gpu_vram_gb()
    vram_category = categorize_vram(vram_gb)
    if vram_gb is not None:
        print(f"\nGPU VRAM: {vram_gb:.2f} GB")
        print(f"VRAM Category: {vram_category}")
    else:
        print("\nGPU VRAM check complete.")


@functools.cache
def is_compile_friendly_gpu(index_or_device: Union[int, str, torch.device] = 0) -> bool:
    """Hand-coded rules from experiments. Don't take seriously."""
    if isinstance(index_or_device, torch.device):
        device = index_or_device
    elif isinstance(index_or_device, str):
        device = torch.device(index_or_device)
    else:
        device = torch.device(get_gpu_type(), index_or_device)

    prop = DeviceProperties.create(device)
    return prop.major >= 8


if __name__ == "__main__":
    main()
    print(is_compile_friendly_gpu("cuda"))
