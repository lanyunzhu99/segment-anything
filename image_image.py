import torch
import requests
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline

# load the pipeline
device = "cuda"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(
    device
)

# let's download an initial image
'''
url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"


response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
'''

init_image = Image.open('/home/lanyunz/sam_2/segment-anything/test_image/3.jpeg').convert("RGB")
init_image.thumbnail((768, 768))

prompt = "A girl walking on the beach, with the same face as the picture"

images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images

images[0].save("./out_image/2.png")