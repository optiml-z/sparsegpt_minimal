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
from dataloader import LatentsDataset
from prompts import prompt
#%%

DATA_PATH = "data"
LOAD = True

if not(LOAD):
    SAVE = True
else:
    SAVE = False

DEBUG = False 

@dataclass
class Args(object):
    nsamples: int = 4
    sparsity = 0.3
    prunen: int = 0
    prunem: int = 0
    percdamp = .01
    blocksize: int = 4
    batch_size: int = 32
    num_layers: int = 5
    input_size: int = 784
    output_size: int = 10
    
args = Args()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)
#%%

# Step 1: Data Preparation
# Define transformations and load datasets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load the training data
calibration_set = LatentsDataset(DATA_PATH, prompt) 
calibration_loader = DataLoader(calibration_set, batch_size=args.batch_size, shuffle=True)


# %%
calibration_set[1]
# %%
from PIL import Image
import torch
from diffusers import AutoencoderKL

device = "cuda"
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True).to(device)
#%%
class SparseGPT:

    def __init__(self, layer):
        self.layer = layer
        print(layer)
        print(type(layer))
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(
        self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prunen == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prunen != 0 and i % prunem == 0:
                    tmp = W1[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                if hasattr(self, 'quantizer'):
                    q = quantize(
                        q.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = W[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()

# %%
@torch.no_grad()
def vae_sequential(model, dataloader, dev):
    print('Starting ...')

    # use_cache = model.config.use_cache
    # model.config.use_cache = False
    
    layers = list(model.modules())[0]
    
    print(layers)
    layers = layers.to(dev)
    layers_dict = find_layers(layers); print(layers_dict)
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.batch_size, 4, 64, 64), dtype=dtype, device=dev
    )

    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError as e:
            print(e)
            pass

    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    # attention_mask = cache['attention_mask']

    print('Ready.')
    gpts = {}
    for i, (layer_name, layer_obj) in enumerate(layers_dict.items()):
        if i == len(layers_dict) - 1:
            break
        layer = layer_obj.to(dev)

        # subset = find_layers(layer)
        
        # gpts = {}
        # for name in subset:
        gpts[layer_name] = SparseGPT(layer_obj)
        print("layer_obj ", layer_obj)
        def add_batch(layer_name):
            def tmp(_, inp, out):
                gpts[layer_name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        
        handles.append(layer_obj.register_forward_hook(add_batch(layer_name)))
        for j in range(args.batch_size):
            # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            outs[j] = layer(inps[j].unsqueeze(0))[0]
        for h in handles:
            h.remove()

        
        print(layer_name)
        print('Pruning ...')
        sparsity = args.sparsity
        gpts[layer_name].fasterprune(
                sparsity, prunen=args.prunen, prunem=args.prunem, percdamp=args.percdamp, blocksize=args.blocksize
            )
        gpts[layer_name].free()

        for j in range(args.batch_size):
            # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            outs[j] = layer(inps[j].unsqueeze(0))[0]

        layer = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps
