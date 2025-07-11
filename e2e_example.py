import argparse
from utils.llm_utils import LLMCodeOptimizer
from prompts import system_prompt, generate_prompt
from utils.pipeline_utils import determine_pipe_loading_memory
from utils.hardware_utils import (
    categorize_vram,
    categorize_ram,
    get_gpu_vram_gb,
    get_system_ram_gb,
    is_compile_friendly_gpu,
    is_fp8_friendly,
)
import torch
from pprint import pprint


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_id",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Can be a repo id from the Hub or a local path where the checkpoint is stored.",
    )
    parser.add_argument(
        "--gemini_model",
        type=str,
        default="gemini-2.5-flash-preview-05-20",
        help="Gemini model to use. Choose from https://ai.google.dev/gemini-api/docs/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="If the `ckpt_id` has variants, supply this flag to estimate compute. Example: 'fp16'.",
    )
    parser.add_argument(
        "--disable_bf16",
        action="store_true",
        help="When enabled the load memory is affected. Prefer not enabling this flag.",
    )
    parser.add_argument(
        "--enable_lossy",
        action="store_true",
        help="When enabled, the code will include snippets for enabling quantization.",
    )
    return parser


def main(args):
    if not torch.cuda.is_available():
        raise ValueError("Not supported for non-CUDA devices for now.")
    
    loading_mem_out = determine_pipe_loading_memory(args.ckpt_id, args.variant, args.disable_bf16)
    load_memory = loading_mem_out["total_loading_memory_gb"]
    ram_gb = get_system_ram_gb()
    ram_category = categorize_ram(ram_gb)
    if ram_gb is not None:
        print(f"\nSystem RAM: {ram_gb:.2f} GB")
        print(f"RAM Category: {ram_category}")
    else:
        print("\nCould not determine System RAM.")

    vram_gb = get_gpu_vram_gb()
    vram_category = categorize_vram(vram_gb)
    if vram_gb is not None:
        print(f"\nGPU VRAM: {vram_gb:.2f} GB")
        print(f"VRAM Category: {vram_category}")
    else:
        print("\nGPU VRAM check complete.")

    is_compile_friendly = is_compile_friendly_gpu()
    is_fp8_compatible = is_fp8_friendly()

    llm = LLMCodeOptimizer(model_name=args.gemini_model, system_prompt=system_prompt)
    current_generate_prompt = generate_prompt.format(
        ckpt_id=args.ckpt_id,
        pipeline_loading_memory=load_memory,
        available_system_ram=ram_gb,
        available_gpu_vram=vram_gb,
        enable_lossy_outputs=args.enable_lossy,
        is_fp8_supported=is_fp8_compatible,
        enable_torch_compile=is_compile_friendly,
    )
    pprint(f"{current_generate_prompt=}")
    print(llm(current_generate_prompt))


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
