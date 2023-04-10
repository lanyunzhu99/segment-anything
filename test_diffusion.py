import torch
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"


pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

prompt = "A picture of a rectangular carpet. This carpet is loved by little girls. This carpet is not a solid color. Its style is very loevely. There are patterns of small animals printed on the carpet. The color of the carpet is a light color with low saturation. The four edges of the carpet have border lines."
prompt = "A carpet in Alice's Wonderland style. This carpet is very cute."
prompt = "A photo of a Chinese beautiful girl with large chest, oval face and large eyes. She wears Jk uniform"
image = pipe(prompt).images[0]  
    
image.save("./diffusion_generate_3.png")
