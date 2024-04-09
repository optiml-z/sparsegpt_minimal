#%%


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
from PIL import Image
from diffusers import AutoencoderKL

device = "cuda"
latents = torch.load("latents.pt").to(device)
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae", use_safetensors=True).to(device)
#%%

def img2cpu(image): 
    image = (image / 2 + 0.5).clamp(0, 1).squeeze() 
    image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    return image


with torch.no_grad():
    for i in range(len(latents)):
        image = vae.decode(latents).sample
        image_cpu = Image.fromarray(img2cpu(image))
        image_cpu.save(f'image_{i:04d}.png')
   

# Start defining a dataloader for the dataset
# %%
