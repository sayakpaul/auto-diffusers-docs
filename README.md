# auto-diffusers-docs

Still a WIP. Use an LLM to generate reasonable code snippets in a hardware-aware manner for Diffusers.

### Motivation

Within the Diffusers, we support a bunch of optimization techniques (refer [here](https://huggingface.co/docs/diffusers/main/en/optimization/memory), [here](https://huggingface.co/docs/diffusers/main/en/optimization/cache), and [here](https://huggingface.co/docs/diffusers/main/en/optimization/fp16)). However, it can be
daunting for our users to determine when to use what. Hence, this repository tries to take a stab
at using an LLM to generate reasonable code snippets for a given pipeline checkpoint that respects
user hardware configuration.

## Getting started

Install the requirements from `requirements.txt`.

Configure `GOOGLE_API_KEY` in the environment: `export GOOGLE_API_KEY=...`.

Then run:

```bash
python e2e_example.py 
```

Full usage:

```sh
usage: e2e_example.py [-h] [--ckpt_id CKPT_ID] [--variant VARIANT] [--enable_lossy]

options:
  -h, --help         show this help message and exit
  --ckpt_id CKPT_ID  Can be a repo id from the Hub or a local path where the checkpoint is stored.
  --variant VARIANT
  --enable_lossy
```

## Example outputs

<details>
<summary>python e2e_example.py (ran on an H100)</summary>

````sh
System RAM: 1999.99 GB
RAM Category: large

GPU VRAM: 79.65 GB
VRAM Category: large
current_generate_prompt='\npipeline_loading_memory_GB: 31.424\navailable_system_ram_GB: 1999.9855346679688\navailable_gpu_vram_GB: 79.6474609375\nenable_lossy_outputs: False\nenable_torch_compile: True\n'
Sending request to Gemini...
```python
from diffusers import DiffusionPipeline
import torch

# User-provided information:
# pipeline_loading_memory_GB: 31.424
# available_system_ram_GB: 1999.9855346679688 (Large RAM)
# available_gpu_vram_GB: 79.6474609375 (Large VRAM)
# enable_lossy_outputs: False
# enable_torch_compile: True

# --- Configuration based on user needs and system capabilities ---

# Placeholder for the actual checkpoint ID
# Please replace this with your desired model checkpoint ID.
CKPT_ID = "black-forest-labs/FLUX.1-dev" 

# Determine dtype. bfloat16 is generally recommended for performance on compatible GPUs.
# Ensure your GPU supports bfloat16 for optimal performance.
dtype = torch.bfloat16

# 1. Pipeline Loading and Device Placement:
# Available VRAM (79.64 GB) is significantly greater than the pipeline's loading memory (31.42 GB).
# Therefore, the entire pipeline can comfortably fit and run on the GPU.
print(f"Loading pipeline '{CKPT_ID}' with {dtype} precision...")
pipe = DiffusionPipeline.from_pretrained(CKPT_ID, torch_dtype=dtype)

print("Moving pipeline to CUDA (GPU) as VRAM is sufficient...")
pipe = pipe.to("cuda")

# 2. Quantization:
# User specified `enable_lossy_outputs: False`, so no quantization is applied.
print("Quantization is NOT applied as per user's preference for lossless outputs.")

# 3. Torch Compile:
# User specified `enable_torch_compile: True`.
# Since no offloading was applied (the entire model is on GPU), we can use `fullgraph=True`
# for potentially greater performance benefits.
print("Applying torch.compile() to the transformer for accelerated inference...")
# The transformer is typically the most compute-intensive part of the diffusion pipeline.
# Compiling it can lead to significant speedups.
pipe.transformer.compile(fullgraph=True)

# --- Inference ---
print("Starting inference...")
prompt = "photo of a dog sitting beside a river, high quality, 4k"
image = pipe(prompt).images[0]

print("Inference completed. Displaying image.")
# Save or display the image
image.save("generated_image.png")
print("Image saved as generated_image.png")

# You can also display the image directly if running in an environment that supports it
# image.show()
```
````
<br>
</details>
<br>
<details>
<summary>python e2e_example.py --enable_lossy</summary>

````sh
System RAM: 1999.99 GB
RAM Category: large

GPU VRAM: 79.65 GB
VRAM Category: large
current_generate_prompt='\npipeline_loading_memory_GB: 31.424\navailable_system_ram_GB: 1999.9855346679688\navailable_gpu_vram_GB: 79.6474609375\nenable_lossy_outputs: True\nenable_torch_compile: True\n'
Sending request to Gemini...
```python
import torch
from diffusers import DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig
import os

# --- User-provided information and derived constants ---
# Checkpoint ID (assuming a placeholder since it was not provided in the user input)
# Using the example CKPT_ID from the problem description
CKPT_ID = "black-forest-labs/FLUX.1-dev"

# Derived from available_gpu_vram_GB (79.64 GB) and pipeline_loading_memory_GB (31.424 GB)
# VRAM is ample to load the entire pipeline
use_cuda_direct_load = True 

# Derived from enable_lossy_outputs (True)
enable_quantization = True

# Derived from enable_torch_compile (True)
enable_torch_compile = True

# --- Inference Code ---

print(f"Loading pipeline: {CKPT_ID}")

# 1. Quantization Configuration (since enable_lossy_outputs is True)
quant_config = None
if enable_quantization:
    # Default to bitsandbytes 4-bit as per guidance
    print("Enabling bitsandbytes 4-bit quantization for 'transformer' component.")
    quant_config = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit", 
        quant_kwargs={"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.bfloat16, "bnb_4bit_quant_type": "nf4"},
        # For FLUX.1-dev, the main generative component is typically 'transformer'.
        # For other pipelines, you might include 'unet', 'text_encoder', 'text_encoder_2', etc.
        components_to_quantize=["transformer"] 
    )

# 2. Load the Diffusion Pipeline
# Use bfloat16 for better performance and modern GPU compatibility
pipe = DiffusionPipeline.from_pretrained(
    CKPT_ID, 
    torch_dtype=torch.bfloat16,
    quantization_config=quant_config if enable_quantization else None
)

# 3. Move Pipeline to GPU (since VRAM is ample)
if use_cuda_direct_load:
    print("Moving the entire pipeline to CUDA (GPU).")
    pipe = pipe.to("cuda")

# 4. Apply torch.compile() (since enable_torch_compile is True)
if enable_torch_compile:
    print("Applying torch.compile() for speedup.")
    # This setting is beneficial when bitsandbytes is used
    torch._dynamo.config.capture_dynamic_output_shape_ops = True 
    
    # Since no offloading is applied (model fits fully in VRAM), use fullgraph=True
    # The primary component for compilation in FLUX.1-dev is 'transformer'
    print("Compiling pipe.transformer with fullgraph=True.")
    pipe.transformer = torch.compile(pipe.transformer, fullgraph=True)

# 5. Perform Inference
print("Starting image generation...")
prompt = "photo of a dog sitting beside a river"
num_inference_steps = 28 # A reasonable number of steps for good quality

# Ensure all inputs are on the correct device for inference after compilation
with torch.no_grad():
    image = pipe(prompt, num_inference_steps=num_inference_steps).images[0]

print("Image generation complete.")
# Save or display the image
output_path = "generated_image.png"
image.save(output_path)
print(f"Image saved to {output_path}")

```
```

</details>