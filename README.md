# auto-diffusers-docs

---
title: "Optimized Diffusers Code"
emoji: 🔥
colorFrom: cyan
colorTo: gray
sdk: gradio
sdk_version: 5.31.0
app_file: app.py
pinned: false
short_description: 'Optimize Diffusers Code on your hardware.'
---

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

By default, the `e2e_example.py` script uses Flux.1-Dev, but this can be configured through the `--ckpt_id` argument.

Full usage:

```sh
usage: e2e_example.py [-h] [--ckpt_id CKPT_ID] [--gemini_model GEMINI_MODEL] [--variant VARIANT] [--enable_lossy]

options:
  -h, --help            show this help message and exit
  --ckpt_id CKPT_ID     Can be a repo id from the Hub or a local path where the checkpoint is stored.
  --gemini_model GEMINI_MODEL
                        Gemini model to use. Choose from https://ai.google.dev/gemini-api/docs/models.
  --variant VARIANT     If the `ckpt_id` has variants, supply this flag to estimate compute. Example: 'fp16'.
  --enable_lossy        When enabled, the code will include snippets for enabling quantization.
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
````

</details>
<br>
When invoked from an RTX 4090, it outputs:

<details>
<summary>Expand</summary>

````sh
System RAM: 125.54 GB
RAM Category: large

GPU VRAM: 23.99 GB
VRAM Category: medium
current_generate_prompt='\npipeline_loading_memory_GB: 31.424\navailable_system_ram_GB: 125.54026794433594\navailable_gpu_vram_GB: 23.98828125\nenable_lossy_outputs: False\nenable_torch_compile: True\n'
Sending request to Gemini...
```python
import torch
from diffusers import DiffusionPipeline
import os # For creating offload directories if needed, though not directly used in this solution

# --- User-provided information (interpreted) ---
# Checkpoint ID will be a placeholder as it's not provided by the user directly in the input.
# pipeline_loading_memory_GB: 31.424 GB
# available_system_ram_GB: 125.54 GB (Categorized as "large": > 40GB)
# available_gpu_vram_GB: 23.98 GB (Categorized as "medium": > 8GB <= 24GB)
# enable_lossy_outputs: False (User prefers no quantization)
# enable_torch_compile: True (User wants to enable torch.compile)

# --- Configuration ---
# Placeholder for the actual checkpoint ID. Replace with the desired model ID.
CKPT_ID = "black-forest-labs/FLUX.1-dev" # Example from Diffusers library.
PROMPT = "photo of a dog sitting beside a river"

print(f"--- Optimizing inference for CKPT_ID: {CKPT_ID} ---")
print(f"Pipeline loading memory: {31.424} GB")
print(f"Available System RAM: {125.54} GB (Large)")
print(f"Available GPU VRAM: {23.98} GB (Medium)")
print(f"Lossy outputs (quantization): {'Disabled' if not False else 'Enabled'}")
print(f"Torch.compile: {'Enabled' if True else 'Disabled'}")
print("-" * 50)

# --- 1. Load the Diffusion Pipeline ---
# Use bfloat16 for a good balance of memory and performance.
print(f"Loading pipeline '{CKPT_ID}' with torch_dtype=torch.bfloat16...")
pipe = DiffusionPipeline.from_pretrained(CKPT_ID, torch_dtype=torch.bfloat16)
print("Pipeline loaded.")

# --- 2. Apply Memory Optimizations ---
# Analysis:
# - Pipeline memory (31.424 GB) exceeds available GPU VRAM (23.98 GB).
# - System RAM (125.54 GB) is large.
# Strategy: Use `enable_model_cpu_offload()`. This moves model components to CPU when not
# in use, swapping them to GPU on demand. This is ideal when VRAM is insufficient but system
# RAM is abundant.

print("Applying memory optimization: `pipe.enable_model_cpu_offload()`...")
pipe.enable_model_cpu_offload()
print("Model CPU offloading enabled. Components will dynamically move between CPU and GPU.")

# --- 3. Apply Speed Optimizations (torch.compile) ---
# Analysis:
# - `enable_torch_compile` is True.
# - Model offloading (`enable_model_cpu_offload`) is applied.
# Strategy: Enable torch.compile with `recompile_limit` as offloading is used.
# Do not use `fullgraph=True` when offloading is active.

print("Applying speed optimization: `torch.compile()`...")
torch._dynamo.config.recompile_limit = 1000 # Recommended when offloading is applied.
# torch._dynamo.config.capture_dynamic_output_shape_ops = True # Only for bitsandbytes, not applicable here.

# Compile the main computational component (e.g., transformer or unet).
# FLUX models primarily use a transformer. For other models, it might be `pipe.unet`.
if hasattr(pipe, "transformer"):
    print("Compiling `pipe.transformer`...")
    pipe.transformer.compile()
elif hasattr(pipe, "unet"):
    print("Compiling `pipe.unet`...")
    pipe.unet.compile()
else:
    print("Warning: Neither `pipe.transformer` nor `pipe.unet` found for compilation. Skipping `torch.compile` for core component.")

print("Speed optimizations applied.")

# --- 4. Perform Inference ---
print(f"Starting image generation for prompt: '{PROMPT}'")
# The pipeline handles component swapping automatically due to `enable_model_cpu_offload()`.
image = pipe(PROMPT).images[0]
print("Image generation complete.")

# --- 5. Save or Display the Result ---
output_path = "generated_image.png"
image.save(output_path)
print(f"Generated image saved to '{output_path}'")

print("\n--- Inference process finished successfully ---")
```
````
</details>

### More outputs

<details>
<summary>"Wan-AI/Wan2.1-T2V-14B-Diffusers" with lossy outputs enabled</summary>

````sh
System RAM: 125.54 GB
RAM Category: large

GPU VRAM: 23.99 GB
VRAM Category: medium
("current_generate_prompt='\\nckpt_id: "
 'Wan-AI/Wan2.1-T2V-14B-Diffusers\\npipeline_loading_memory_GB: '
 '37.432\\navailable_system_ram_GB: '
 '125.54026794433594\\navailable_gpu_vram_GB: '
 '23.98828125\\nenable_lossy_outputs: True\\nis_fp8_supported: '
 "True\\nenable_torch_compile: True\\n'")
Sending request to Gemini...
```python
from diffusers import DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig
import torch

ckpt_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"

quant_config = PipelineQuantizationConfig(
    quant_backend="torchao",
    quant_kwargs={"quant_type": "float8dq_e4m3_row"},
    components_to_quantize=["transformer"]
)
pipe = DiffusionPipeline.from_pretrained(ckpt_id, quantization_config=quant_config, torch_dtype=torch.bfloat16)

# Apply model CPU offload due to VRAM constraints
pipe.enable_model_cpu_offload()

# torch.compile() configuration
torch._dynamo.config.recompile_limit = 1000
pipe.transformer.compile()
# pipe.vae.decode = torch.compile(pipe.vae.decode) # Uncomment if you want to compile VAE decode as well

prompt = "photo of a dog sitting beside a river"

# Modify the pipe call arguments as needed.
image = pipe(prompt).images[0]

# You can save the image or perform further operations here
# image.save("generated_image.png")
```
````
</details>
<small>Ran on an RTX 4090</small>