import collections
from pathlib import Path
import functools
import os
import safetensors.torch
from huggingface_hub import model_info
import tempfile
import torch
import functools
import os
import requests
import struct
from huggingface_hub import hf_hub_url

DTYPE_MAP = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}


# https://huggingface.co/docs/safetensors/v0.3.2/metadata_parsing#python
def _parse_single_file(url):
    print(f"{url=}")
    token = os.getenv("HF_TOKEN")
    assert token, "HF_TOKEN must be set"
    headers = {"Range": "bytes=0-7", "Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    length_of_header = struct.unpack("<Q", response.content)[0]
    headers = {"Range": f"bytes=8-{7 + length_of_header}", "Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    header = response.json()
    return header


def _get_dtype_from_safetensor_file(file_path):
    """Inspects a safetensors file and returns the dtype of the first tensor.

    If it's not a safetensors file and a URL instead, we query it.
    """
    if "https" in file_path:
        metadata = _parse_single_file(file_path)
        except_format_metadata_keys = sorted({k for k in metadata if k != "__metadata__"})
        string_dtype = metadata[except_format_metadata_keys[0]]["dtype"]
        return DTYPE_MAP[string_dtype]
    try:
        # load_file is simple and sufficient for this info-gathering purpose.
        state_dict = safetensors.torch.load_file(file_path)
        if not state_dict:
            return "N/A (empty)"

        # Get the dtype from the first tensor in the state dict
        first_tensor = next(iter(state_dict.values()))
        return first_tensor.dtype
    except Exception as e:
        print(f"Warning: Could not determine dtype from {file_path}. Error: {e}")
        return "N/A (error)"


def _process_components(component_files, file_accessor_fn, disable_bf16=False):
    """
    Generic function to process components, calculate size, and determine dtype.

    Args:
        component_files (dict): A dictionary mapping component names to lists of file objects.
        file_accessor_fn (function): A function that takes a file object and returns
                                     a tuple of (local_path_for_inspection, size_in_bytes, relative_filename).
        disable_bf16 (bool): To disable using `torch.bfloat16`. Use it at your own risk.

    Returns:
        dict: A dictionary containing the total memory and detailed component info.
    """
    components_info = {}
    total_size_bytes = 0

    for name, files in component_files.items():
        # Get dtype by inspecting the first file of the component
        first_file = files[0]

        # The accessor function handles how to get the path (download vs local)
        # and its size and relative name.
        inspection_path, _, _ = file_accessor_fn(first_file)
        dtype = _get_dtype_from_safetensor_file(inspection_path)

        component_size_bytes = 0
        component_file_details = []
        for f in files:
            _, size_bytes, rel_filename = file_accessor_fn(f)
            component_size_bytes += size_bytes
            component_file_details.append({"filename": rel_filename, "size_mb": size_bytes / (1024**2)})

        if dtype == torch.float32 and not disable_bf16:
            print(
                f"The `dtype` for component ({name}) is torch.float32. Since bf16 computation is not disabled "
                "we will slash the total size of this component by 2."
            )
            total_size_bytes += component_size_bytes / 2
        else:
            total_size_bytes += component_size_bytes

        components_info[name] = {
            "size_gb": round(component_size_bytes / (1024**3), 3),
            "dtype": dtype,
            "files": sorted(component_file_details, key=lambda x: x["filename"]),
        }

    return {
        "total_loading_memory_gb": round(total_size_bytes / (1024**3), 3),
        "components": components_info,
    }


@functools.lru_cache()
def _determine_memory_from_hub_ckpt(ckpt_id, variant=None, disable_bf16=False):
    """
    Determines memory and dtypes for a checkpoint on the Hugging Face Hub.
    """
    files_in_repo = model_info(ckpt_id, files_metadata=True, token=os.getenv("HF_TOKEN")).siblings
    all_safetensors_siblings = [
        s for s in files_in_repo if s.rfilename.endswith(".safetensors") and "/" in s.rfilename
    ]
    if variant:
        all_safetensors_siblings = [f for f in all_safetensors_siblings if variant in f.rfilename]

    component_files = collections.defaultdict(list)
    for sibling in all_safetensors_siblings:
        component_name = Path(sibling.rfilename).parent.name
        component_files[component_name].append(sibling)

    with tempfile.TemporaryDirectory() as temp_dir:

        def hub_file_accessor(file_obj):
            """Accessor for Hub files: downloads them and returns path/size."""
            print(f"Querying '{file_obj.rfilename}' for inspection...")
            url = hf_hub_url(ckpt_id, file_obj.rfilename)
            return url, file_obj.size, file_obj.rfilename

        # We only need to download one file per component for dtype inspection.
        # To make this efficient, we create a specialized accessor for the processing loop
        # that only downloads the *first* file encountered for a component.
        downloaded_for_inspection = {}

        def efficient_hub_accessor(file_obj):
            component_name = Path(file_obj.rfilename).parent.name
            if component_name not in downloaded_for_inspection:
                path, _, _ = hub_file_accessor(file_obj)
                downloaded_for_inspection[component_name] = path

            inspection_path = downloaded_for_inspection[component_name]
            return inspection_path, file_obj.size, file_obj.rfilename

        return _process_components(component_files, efficient_hub_accessor, disable_bf16)


@functools.lru_cache()
def _determine_memory_from_local_ckpt(path: str, variant=None, disable_bf16=False):
    """
    Determines memory and dtypes for a local checkpoint.
    """
    ckpt_path = Path(path)
    if not ckpt_path.is_dir():
        return {"error": f"Checkpoint path '{path}' not found or is not a directory."}

    all_safetensors_paths = list(ckpt_path.glob("**/*.safetensors"))
    if variant:
        all_safetensors_paths = [p for p in all_safetensors_paths if variant in p.name]

    component_files = collections.defaultdict(list)
    for file_path in all_safetensors_paths:
        component_name = file_path.parent.name
        component_files[component_name].append(file_path)

    def local_file_accessor(file_path):
        """Accessor for local files: just returns their path and size."""
        return file_path, file_path.stat().st_size, str(file_path.relative_to(ckpt_path))

    return _process_components(component_files, local_file_accessor, disable_bf16)


def determine_pipe_loading_memory(ckpt_id: str, variant=None, disable_bf16=False):
    """
    Determines the memory and dtypes for a pipeline, whether it's local or on the Hub.
    """
    if os.path.isdir(ckpt_id):
        return _determine_memory_from_local_ckpt(ckpt_id, variant, disable_bf16)
    else:
        return _determine_memory_from_hub_ckpt(ckpt_id, variant, disable_bf16)


if __name__ == "__main__":
    output = _determine_memory_from_hub_ckpt("Wan-AI/Wan2.1-T2V-14B-Diffusers")
    total_size_gb = output["total_loading_memory_gb"]
    safetensor_files = output["components"]
    print(f"{total_size_gb=} GB")
    print(f"{safetensor_files=}")
    print("\n")