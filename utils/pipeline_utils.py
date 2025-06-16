from huggingface_hub import model_info
from pathlib import Path
import functools
import os


@functools.lru_cache()
def _determine_memory_from_hub_ckpt(ckpt_id, variant=None):
    """
    Determine loading memory requirements of the `.safetensors` files given a Diffusers-style
    checkpoint.
    """
    files_in_repo = model_info(ckpt_id, files_metadata=True).siblings
    # "/": because diffusers is always folder-style.
    all_safetensors_siblings = [
        sibling for sibling in files_in_repo if sibling.rfilename.endswith(".safetensors") and "/" in sibling.rfilename
    ]
    if variant:
        all_safetensors_siblings = [f for f in all_safetensors_siblings if variant in f.rfilename]
    total_size_bytes = 0
    safetensors_files = []

    for sibling in all_safetensors_siblings:
        file_size_bytes = sibling.size
        size_mb = file_size_bytes / (1024 * 1024)
        safetensors_files.append(
            {"filename": sibling.rfilename, "size_mb": size_mb, "estimated": file_size_bytes != sibling.size}
        )
        total_size_bytes += file_size_bytes

    total_size_gb = round(total_size_bytes / (1024**3), 3)
    safetensors_files = sorted(safetensors_files, key=lambda item: item["filename"])
    return total_size_gb, safetensors_files


@functools.lru_cache()
def _determine_memory_from_local_ckpt(path: str, variant=None):
    ckpt_path = Path(path)
    if not ckpt_path.is_dir():
        print(f"Error: Checkpoint path '{path}' not found or is not a directory.")
        return 0.0, []

    all_safetensors_siblings = list(ckpt_path.glob("**/*.safetensors"))

    if variant:
        all_safetensors_siblings = [f for f in all_safetensors_siblings if variant in f.name]

    total_size_bytes = 0
    safetensors_files = []
    for file_path in all_safetensors_siblings:
        try:
            file_size_bytes = file_path.stat().st_size
            size_mb = file_size_bytes / (1024 * 1024)
            safetensors_files.append(
                {
                    "filename": str(file_path.relative_to(ckpt_path)),
                    "size_mb": size_mb,
                    "estimated": False,  # For local files, the size is exact.
                }
            )
            total_size_bytes += file_size_bytes
        except FileNotFoundError:
            print(f"Warning: File '{file_path}' was found but could not be accessed.")
            continue

    total_size_gb = round(total_size_bytes / (1024**3), 3)
    safetensors_files = sorted(safetensors_files, key=lambda item: item["filename"])
    return total_size_gb, safetensors_files


def determine_pipe_loading_memory(ckpt_id: str, variant=None):
    if os.path.exists(ckpt_id) and os.path.isdir(ckpt_id):
        return _determine_memory_from_local_ckpt(ckpt_id, variant)
    else:
        return _determine_memory_from_hub_ckpt(ckpt_id, variant)


if __name__ == "__main__":
    total_size_gb, safetensor_files = _determine_memory_from_hub_ckpt("black-forest-labs/FLUX.1-dev")
    print(f"{total_size_gb=} GB")
    print(f"{safetensor_files=}")
    print("\n")
    # total_size_gb, safetensor_files = _determine_memory_from_local_ckpt("LOCAL_DIR") # change me.
    # print(f"{total_size_gb=} GB")
    # print(f"{safetensor_files=}")
