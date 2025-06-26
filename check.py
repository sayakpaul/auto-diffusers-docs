from utils.pipeline_utils import _parse_single_file

res = _parse_single_file(
    "black-forest-labs/FLUX.1-dev", "transformer/diffusion_pytorch_model-00001-of-00003.safetensors"
)

except_format_metadata_keys = sorted({k for k in res if k != "__metadata__"})
print(res[except_format_metadata_keys[0]])
