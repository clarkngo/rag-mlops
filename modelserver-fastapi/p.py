from diffusers import DiffusionPipeline
import torch
from PIL import Image
from io import BytesIO

# Initialize the pipeline
pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
pipeline.load_lora_weights("kviai/3d-icons")
pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

def generate_image(prompt: str) -> Image.Image:
    """Generate an image based on the given prompt."""
    # Generate image using the pipeline
    result = pipeline(prompt=prompt)
    return result.images[0]

def save_image(image: Image.Image, filename: str):
    """Save the image to a file."""
    image.save(filename, format="PNG")

if __name__ == "__main__":
    prompt = input("Enter a prompt for image generation: ")
    try:
        # Generate the image
        image = generate_image(prompt)
        
        # Save the image
        output_filename = "generated_image.png"
        save_image(image, output_filename)
        
        print(f"Image successfully generated and saved as {output_filename}")
    except Exception as e:
        print(f"An error occurred: {e}")
