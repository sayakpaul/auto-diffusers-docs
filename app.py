import gradio as gr
from utils.pipeline_utils import determine_pipe_loading_memory
from utils.llm_utils import LLMCodeOptimizer
from prompts import system_prompt, generate_prompt
from utils.hardware_utils import categorize_ram, categorize_vram

LLM_CACHE = {}


def get_output_code(
    repo_id,
    gemini_model_to_use,
    disable_bf16,
    enable_lossy,
    system_ram,
    gpu_vram,
    torch_compile_friendly,
    fp8_friendly,
):
    loading_mem_out = determine_pipe_loading_memory(repo_id, None, disable_bf16)
    load_memory = loading_mem_out["total_loading_memory_gb"]
    ram_category = categorize_ram(system_ram)
    vram_category = categorize_vram(gpu_vram)

    print(f"RAM Category: {ram_category}")
    print(f"VRAM Category: {vram_category}")

    if gemini_model_to_use not in LLM_CACHE:
        print(f"Initializing new LLM instance for: {gemini_model_to_use}")
        # If not, create it and add it to the cache
        LLM_CACHE[gemini_model_to_use] = LLMCodeOptimizer(model_name=gemini_model_to_use, system_prompt=system_prompt)

    llm = LLM_CACHE[gemini_model_to_use]
    current_generate_prompt = generate_prompt.format(
        ckpt_id=repo_id,
        pipeline_loading_memory=load_memory,
        available_system_ram=system_ram,
        available_gpu_vram=gpu_vram,
        enable_lossy_outputs=enable_lossy,
        is_fp8_supported=fp8_friendly,
        enable_torch_compile=torch_compile_friendly,
    )
    generated_prompt = current_generate_prompt
    llm_output = llm(current_generate_prompt)
    return llm_output, generated_prompt


# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🧨 Generate Diffusers Inference code snippet tailored to your machine
        Enter a Hugging Face Hub `repo_id` and your system specs to get started for inference.
        This tool uses [Gemini](https://ai.google.dev/gemini-api/docs/models) to generate the code based on your settings. This is based on
        [sayakpaul/auto-diffusers-docs](https://github.com/sayakpaul/auto-diffusers-docs/).
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            repo_id = gr.Textbox(
                label="Hugging Face Repo ID",
                placeholder="e.g., black-forest-labs/FLUX.1-dev",
                info="The model repository you want to analyze.",
                value="black-forest-labs/FLUX.1-dev",
            )
            gemini_model_to_use = gr.Dropdown(
                ["gemini-2.5-flash", "gemini-2.5-pro"],
                value="gemini-2.5-flash",
                label="Gemini Model",
                info="Select the model to generate the analysis.",
            )
            with gr.Row():
                system_ram = gr.Number(label="System RAM (GB)", value=20)
                gpu_vram = gr.Number(label="GPU VRAM (GB)", value=8)

            with gr.Row():
                disable_bf16 = gr.Checkbox(
                    label="Disable BF16 (Use FP32)",
                    value=False,
                    info="Calculate using 32-bit precision instead of 16-bit.",
                )
                enable_lossy = gr.Checkbox(
                    label="Allow Lossy Quantization", value=False, info="Consider 8-bit/4-bit quantization."
                )
                torch_compile_friendly = gr.Checkbox(
                    label="torch.compile() friendly", value=False, info="Model is compatible with torch.compile."
                )
                fp8_friendly = gr.Checkbox(
                    label="fp8 friendly", value=False, info="Model and hardware support FP8 precision."
                )

        with gr.Column(scale=1):
            submit_btn = gr.Button("Estimate Memory ☁", variant="primary", scale=1)

    with gr.Accordion("Generated LLM Prompt (for debugging)", open=False):
        prompt_output = gr.Textbox(label="Prompt", show_copy_button=True, lines=10, interactive=False)

    gr.Markdown("---")
    gr.Markdown("### Estimation Result")

    output_markdown = gr.Markdown(label="LLM Output", value="*Your results will appear here...*")

    gr.Markdown(
        """
        ---
        > ⛔️ **Disclaimer:** Large Language Models (LLMs) can make mistakes. The information provided
        > is an estimate and should be verified. Always test the model on your target hardware to confirm
        > actual memory requirements.
        """
    )

    # --- Event Handling ---
    all_inputs = [
        repo_id,
        gemini_model_to_use,
        disable_bf16,
        enable_lossy,
        system_ram,
        gpu_vram,
        torch_compile_friendly,
        fp8_friendly,
    ]
    submit_btn.click(fn=get_output_code, inputs=all_inputs, outputs=[output_markdown, prompt_output])


if __name__ == "__main__":
    demo.launch()
