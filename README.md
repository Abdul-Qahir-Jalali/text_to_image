# Text-to-Image Generator

This project is a simple **Text-to-Image Generator** that uses **Stable Diffusion** and **Gradio**.

You can type any prompt (like "a sunset over the mountains"), and the model will generate an image based on your description.

## Demo

(Coming soon on Hugging Face Spaces!)

## How it works

- The app uses the `runwayml/stable-diffusion-v1-5` model from Hugging Face.
- The model is loaded using the `diffusers` library.
- The interface is created with `gradio` for easy interaction.

## Installation

1. Install the required libraries:

```bash
pip install torch diffusers gradio
