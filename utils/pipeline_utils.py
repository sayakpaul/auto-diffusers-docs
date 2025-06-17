from huggingface_hub import model_info
from pathlib import Path
import functools
import os


from huggingface_hub import model_info
from pathlib import Path
import functools
import os

def _calculate_memory_from_iterator(files_iterator):
    """
    A generic helper function to calculate total size from an iterator of files.

    Args:
        files_iterator: An iterator that yields tuples of
                        (filename, size_in_bytes, is_estimated_flag).

    Returns:
        A tuple containing (total_size_gb, list_of_file_details).
    """
    total_size_bytes = 0
    safetensors_files = []

    for filename, file_size_bytes, estimated in files_iterator:
        size_mb = file_size_bytes / (1024 * 1024)
        safetensors_files.append(
            {"filename": filename, "size_mb": size_mb, "estimated": estimated}
        )
        total_size_bytes += file_size_bytes

    total_size_gb = round(total_size_bytes / (1024**3), 3)
    # Sort the final list by filename for consistent output
    safetensors_files = sorted(safetensors_files, key=lambda item: item["filename"])
    return total_size_gb, safetensors_files


@functools.lru_cache()
def _determine_memory_from_hub_ckpt(ckpt_id, variant=None):
    """
    Determine loading memory requirements for a checkpoint on the Hugging Face Hub.
    """
    files_in_repo = model_info(ckpt_id, files_metadata=True).siblings
    # Diffusers checkpoints are always in a folder structure
    all_safetensors_siblings = [
        sibling for sibling in files_in_repo if sibling.rfilename.endswith(".safetensors") and "/" in sibling.rfilename
    ]
    if variant:
        all_safetensors_siblings = [f for f in all_safetensors_siblings if variant in f.rfilename]

    def hub_files_iterator():
        """Generator to yield file info in a standardized format."""
        for sibling in all_safetensors_siblings:
            # The original logic `sibling.size != sibling.size` is always False.
            # This is kept for consistency, but in a real-world scenario,
            # you might have a better way to determine if a size is estimated.
            yield (sibling.rfilename, sibling.size, sibling.size != sibling.size)

    return _calculate_memory_from_iterator(hub_files_iterator())


@functools.lru_cache()
def _determine_memory_from_local_ckpt(path: str, variant=None):
    """
    Determine loading memory requirements for a local checkpoint.
    """
    ckpt_path = Path(path)
    if not ckpt_path.is_dir():
        print(f"Error: Checkpoint path '{path}' not found or is not a directory.")
        return 0.0, []

    all_safetensors_siblings = list(ckpt_path.glob("**/*.safetensors"))

    if variant:
        all_safetensors_siblings = [f for f in all_safetensors_siblings if variant in f.name]

    def local_files_iterator():
        """Generator to yield file info in a standardized format."""
        for file_path in all_safetensors_siblings:
            try:
                file_size_bytes = file_path.stat().st_size
                # For local files, the size is always exact.
                yield (str(file_path.relative_to(ckpt_path)), file_size_bytes, False)
            except FileNotFoundError:
                print(f"Warning: File '{file_path}' was found but could not be accessed during size check.")
                continue

    return _calculate_memory_from_iterator(local_files_iterator())


def determine_pipe_loading_memory(ckpt_id: str, variant=None):
    """
    Determines the memory required to load a pipeline, whether it's local or on the Hub.
    """
    if os.path.isdir(ckpt_id):
        return _determine_memory_from_local_ckpt(ckpt_id, variant)
    else:
        # Assumes it's a Hugging Face Hub repository ID if not a local directory
        return _determine_memory_from_hub_ckpt(ckpt_id, variant)


if __name__ == "__main__":
    total_size_gb, safetensor_files = _determine_memory_from_hub_ckpt("black-forest-labs/FLUX.1-dev")
    print(f"{total_size_gb=} GB")
    print(f"{safetensor_files=}")
    print("\n")
    # total_size_gb, safetensor_files = _determine_memory_from_local_ckpt("LOCAL_DIR") # change me.
    # print(f"{total_size_gb=} GB")
    # print(f"{safetensor_files=}")
