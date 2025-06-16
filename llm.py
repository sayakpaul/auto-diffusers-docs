import google.generativeai as genai
import os

# https://ai.google.dev/gemini-api/docs/models#gemini-2.5-flash-preview
MODEL_NAME = "gemini-2.5-flash-preview-05-20"


class LLMCodeOptimizer:
    def __init__(self, system_prompt: str):
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key is None:
            raise ValueError("Must provide an API key for Gemini through the `GOOGLE_API_KEY` env variable.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=system_prompt)

    def __call__(self, generation_prompt):
        try:
            print("Sending request to Gemini...")
            response = self.model.generate_content(generation_prompt)

            return response.text

        except Exception as e:
            # Handle potential exceptions, such as invalid API keys,
            # network issues, or content moderation errors.
            return f"An error occurred: {e}"
