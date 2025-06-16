import argparse
from llm import LLMCodeOptimizer
from prompts import system_prompt, generate_prompt
from pipeline_utils import determine_pipe_loading_memory
from hardware_utils import categorize_vram, categorize_ram, get_gpu_vram_gb, get_system_ram_gb, is_compile_friendly_gpu


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_id",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Can be a repo id from the Hub or a local path where the checkpoint is stored.",
    )
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--enable_lossy", action="store_true")
    return parser


def main(args):
    load_memory = determine_pipe_loading_memory(args.ckpt_id, args.variant)[0]

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

    llm = LLMCodeOptimizer(system_prompt=system_prompt)
    current_generate_prompt = generate_prompt.format(
        pipeline_loading_memory=load_memory,
        available_system_ram=ram_gb,
        available_gpu_vram=vram_gb,
        enable_lossy_outputs=args.enable_lossy,
        enable_torch_compile=is_compile_friendly,
    )
    print(f"{current_generate_prompt=}")
    print(llm(current_generate_prompt))


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
