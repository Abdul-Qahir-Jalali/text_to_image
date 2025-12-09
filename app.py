import torch
from diffusers import StableDiffusionPipeline
import gradio as gr
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model on CPU
# Use a cached path or standard load. HF Spaces usually handles caching automatically.
try:
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        use_safetensors=True
    ).to("cpu")
    # Optimize for CPU if available (optional, but good for speed)
    # pipe.enable_attention_slicing() 
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise e

# Define function to generate image
def generate_image(prompt):
    try:
        logger.info(f"Generating image for prompt: {prompt}")
        with torch.no_grad():
            image = pipe(prompt, num_inference_steps=20).images[0] # Reduced steps for faster CPU generation
        return image
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return None

# Custom CSS
custom_css = """
body {
    background: linear-gradient(to right, #00c6ff, #0072ff) !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}
h1 {
    color: white;
    text-align: center;
    margin-top: 20px;
    font-size: 3em;
}
footer {
    text-align: center;
    color: white;
    margin-top: 50px;
}
.gr-button {
    background: #ff7e5f !important;
    color: white !important;
    font-weight: bold !important;
    border-radius: 10px !important;
}
"""

# Gradio Interface
with gr.Blocks(css=custom_css, title="Text-to-Image Generator") as iface:
    gr.HTML("<h1>🎨 Text-to-Image Generator by Qahir 🎨</h1>")

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Enter your imagination ✨", 
                placeholder="e.g., A futuristic city in the sky...", 
                lines=2
            )
            generate_button = gr.Button("🚀 Generate Image", elem_classes=["gr-button"])
        with gr.Column(scale=2):
            output_image = gr.Image(label="Generated Image", type="pil", height=400)

    generate_button.click(fn=generate_image, inputs=prompt, outputs=output_image)

    gr.HTML("<footer>Made with ❤️ by Qahir</footer>")

# Launch
if __name__ == "__main__":
    iface.launch()
