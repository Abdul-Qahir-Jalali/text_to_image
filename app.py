import torch
from diffusers import StableDiffusionPipeline
import gradio as gr

# Load model on CPU
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
).to("cpu")

# Define function to generate image
def generate_image(prompt):
    with torch.no_grad():
        image = pipe(prompt).images[0]
    return image

# Custom CSS
custom_css = """
body {
    background: linear-gradient(to right, #00c6ff, #0072ff);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
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
    background: #ff7e5f;
    color: white;
    font-weight: bold;
    border-radius: 10px;
}
"""

# Gradio Interface
with gr.Blocks(css=custom_css) as iface:
    gr.HTML("<h1>🎨 Text-to-Image Generator by Qahir 🎨</h1>")

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="Enter your imagination ✨", placeholder="e.g., A futuristic city in the sky...", lines=2)
            generate_button = gr.Button("🚀 Generate Image")
        with gr.Column(scale=2):
            output_image = gr.Image(label="Generated Image", height=400)

    generate_button.click(fn=generate_image, inputs=prompt, outputs=output_image)

    gr.HTML("<footer>Made with ❤️ by Qahir</footer>")

# Launch
if __name__ == "__main__":
    iface.launch()
