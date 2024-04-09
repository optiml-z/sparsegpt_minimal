import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from modelutils import *
from quant import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dataclasses import dataclass
from torch.utils.data import SubsetRandomSampler



DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae", use_safetensors=True)
tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="text_encoder", use_safetensors=True
)
unet = UNet2DConditionModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="unet", use_safetensors=True
)


from diffusers import UniPCMultistepScheduler

scheduler = UniPCMultistepScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

torch_device = "cuda"
vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device);

# prompt = ["a photograph of an astronaut riding a horse"]
prompt = [
    "A futuristic cityscape at dusk, illuminated by neon lights and flying cars, in the style of cyberpunk.",
    "A serene mountain landscape with a crystal-clear lake in the foreground, during the golden hour.",
    "A bustling medieval marketplace, full of vibrant colors and characters, with a castle in the background.",
    "An underwater scene showing a coral reef teeming with life, including a variety of fish and a sunken ship.",
    "A surreal landscape where the sky is made of swirling galaxies and the ground is a checkerboard of grass and clouds.",
    "A portrait of a Victorian lady, detailed lace dress and an intricate hairstyle, holding a mysterious locket.",
    "A close-up of a dragon's eye, reflecting a knight preparing for battle, with scales shimmering in the sunlight.",
    "An abandoned amusement park overtaken by nature, with a vintage carousel in the foreground.",
    "A whimsical forest inhabited by fantastical creatures, with a glowing path leading to an ancient tree.",
    "A dystopian cityscape showing the contrast between a wealthy district and a rundown area, in a high-tech future.",
    "A traditional Japanese garden in spring, with cherry blossoms in full bloom and a tranquil koi pond.",
    "An art deco style poster advertising a luxurious 1920s ocean liner, with elegant fonts and geometric shapes.",
    "A steampunk workshop filled with intricate machinery, gears, and a work-in-progress invention.",
    "A snowy landscape at night lit by the aurora borealis, with a cozy wooden cabin and a smoking chimney.",
    "A detailed map of a fantasy world, featuring diverse terrains, kingdoms, and mythical landmarks.",
    "A space station orbiting an alien planet, with spaceships docking and astronauts performing a spacewalk.",
    "A Renaissance-era banquet scene, with sumptuously dressed figures, an opulent table setting, and a grand hall.",
    "A post-apocalyptic city with nature reclaiming the ruins, and a group of survivors exploring.",
    "A magical library with floating books, ancient tomes, and a glowing portal to another dimension.",
    "A scene from ancient Egypt, showing the construction of the pyramids with workers and Pharaoh overseeing.",
    "A high-speed chase scene through a futuristic metropolis, with advanced vehicles and neon lights.",
    "A cozy autumn scene in a small village, with leaves falling, a pumpkin patch, and a warm bakery.",
    "An explorer discovering an ancient ruin in a jungle, with hidden traps and treasures.",
    "A deep space scene showing a nebula, distant stars, and a solitary spaceship on an exploration mission.",
    "A magical girl transformation scene, with dynamic poses, sparkling effects, and a cute mascot.",
    "A noir-style detective scene, with a shadowy figure, a vintage office, and a case waiting to be solved.",
    "A Viking longship sailing through a stormy sea, with lightning illuminating the fearsome warriors.",
    "A cybernetic samurai showdown in a neon-lit Tokyo alley, with futuristic armor and weapons.",
    "A peaceful Zen garden with smooth stones, raked sand patterns, and a small bamboo fountain.",
    "An epic battle scene from a fantasy novel, with dragons, wizards, and warriors clashing.",
    "A vintage circus poster featuring an exotic animal act, with bold typography and classic illustrations.",
    "A mystical encounter in an enchanted forest, with a unicorn and a fairy under a full moon."
]

height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
num_inference_steps = 25  # Number of denoising steps
guidance_scale = 7.5  # Scale for classifier-free guidance
generator = torch.Generator(device = torch_device).manual_seed(0) # Seed generator to create the initial latent noise
batch_size = len(prompt)

text_input = tokenizer(
    prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
)

with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
    
max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

latents = torch.randn(
    (batch_size, unet.config.in_channels, height // 8, width // 8),
    generator=generator,
    device=torch_device,
)

latents = latents * scheduler.init_noise_sigma

from tqdm.auto import tqdm

scheduler.set_timesteps(num_inference_steps)

for t in tqdm(scheduler.timesteps):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

    # predict the noise residual
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, t, latents).prev_sample
    
# scale and decode the image latents with vae
latents = 1 / 0.18215 * latents
torch.save(latents, 'latents.pt')