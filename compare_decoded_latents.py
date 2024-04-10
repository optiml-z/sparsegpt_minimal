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
from dataloader import LatentsDataset
from prompts import prompt
import os
#%%


data_path = ".\data"
device = "cuda"
#%%
dataset = LatentsDataset(data_path, prompt)

#%%
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
#%%
dataloader
#%%
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae", use_safetensors=True).to(device)
#%%

def img2cpu(image): 
    image = (image / 2 + 0.5).clamp(0, 1).squeeze() 
    image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    return image

if not os.path.exists('images'):
    os.makedirs('images')

with torch.no_grad():
    for i in range(len(dataset)):
        image = vae.decode(dataset[i]).sample
        image_cpu = Image.fromarray(img2cpu(image))
        image_cpu.save(os.path.join("images", f'image_{i}.png'))
   

# Start defining a dataloader for the dataset
# %%
