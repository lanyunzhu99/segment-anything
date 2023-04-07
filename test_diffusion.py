import torch
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"


pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

prompt = "A girl is playing her guitar in the woods by the creek, a jeep is parked next to the girl"
image = pipe(prompt).images[0]  
    
image.save("./diffusion_generate_2.png")
