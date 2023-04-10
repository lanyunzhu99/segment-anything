import torch
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"


pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

prompt = "A picture of a rectangular rug. This rug is not a solid color, it has a cute style with small animal prints on the rug, the color of the rug is a lighter color with less saturation, and the four sides of the rug have border lines."
image = pipe(prompt).images[0]  
    
image.save("./diffusion_generate_3.png")
