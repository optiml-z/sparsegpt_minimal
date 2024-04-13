# %%
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

# %%
MODEL_NAME = "dynamic_mnist_mlp.pth"
LOAD = True

if not(LOAD):
    SAVE = True
else:
    SAVE = False

DEBUG = True 

@dataclass
class Args(object):
    nsamples: int = 8
    sparsity = 0.3
    prunen: int = 0
    prunem: int = 0
    percdamp = .01
    blocksize: int = 4
    batch_size: int = 8
    num_layers: int = 5
    input_size: int = 784
    output_size: int = 10
    
args = Args()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)


# %%
# Step 1: Data Preparation
# Define transformations and load datasets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

# Download and load the test data
testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

# Confirm if data is loaded
len(trainset), len(testset)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')


# %%
class DynamicMLP(nn.Module):
    def __init__(self, input_size=28*28, output_size=10, hidden_sizes=0, num_layers=5):
        super(DynamicMLP, self).__init__()
        
        # Ensure hidden_sizes is a list for consistency
        if not isinstance(hidden_sizes, list):
            hidden_sizes = [input_size] * num_layers
        elif len(hidden_sizes) == 0:
            hidden_sizes += [input_size] * (num_layers - len(hidden_sizes))
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        # Hidden layers
        for i in range(1, num_layers):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[min(i, len(hidden_sizes)-1)]))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # ModuleList to hold all layers
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        # Apply each layer with ReLU activation, except for the output layer
        x = x.view(-1, 28*28) # Flatten the image        
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        # Output layer without ReLU activation
        x = self.layers[-1](x)
        return x
    
class MLPSeq(nn.Module):
    def __init__(self):
        super(MLPSeq, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(28**2, 28**2) for i in range(5)])        
        # self.linears.insert(0, nn.Flatten())
        # self.fc_block1 = nn.Sequential(
        #     nn.Linear(28*28, 28*28),
        #     nn.ReLU(),
        #     nn.Linear(28*28, 28*28),
        #     nn.ReLU(),
        # )
        # self.fc_block2 = nn.Sequential(
        #     nn.Linear(28*28, 28*28),
        #     nn.ReLU(),
        #     nn.Linear(28*28, 28*28),
        #     nn.ReLU(),
        # )
        self.out = nn.Linear(28*28, 10)        

    def forward(self, x):
        x = x.view(-1, 28*28) # Flatten the image
        for layer in self.linears:
            x = torch.relu(layer(x))
        # x = self.fc_block1(x)
        # x = self.fc_block2(x)
        x = self.out(x)
        return x

# %%
model = DynamicMLP(num_layers=args.num_layers).to(device)
# model = MLPSeq().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 1
train_losses, test_losses = [], []

# %%
if LOAD:
    PATH = f'./{MODEL_NAME}'
    model.load_state_dict(torch.load(PATH))
else:    
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            optimizer.zero_grad()
            output = model(images.to(device))
            loss = criterion(output, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            test_loss = 0
            accuracy = 0
            with torch.no_grad():
                for images, labels in testloader:
                    log_ps = model(images.to(device))
                    test_loss += criterion(log_ps, labels.to(device))
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.to(device).view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
                    
            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))

            print(f"Epoch {e+1}/{epochs}.. "
                f"Train loss: {running_loss/len(trainloader):.3f}.. "
                f"Test loss: {test_loss/len(testloader):.3f}.. "
                f"Test accuracy: {accuracy/len(testloader):.3f}")

    print('Finished Training')
    if SAVE:
        PATH = f'./{MODEL_NAME}'
        torch.save(model.state_dict(), PATH)
        plt.plot(train_losses, label='Training loss')
        plt.plot(list(map(torch.Tensor.cpu, test_losses)), label='Validation loss')
        plt.legend(frameon=False)
        plt.show()


# %%
def infer_and_compute_accuracy_random_samples(model, dataset, num_samples=1000):
    model.eval()  # Set the model to evaluation mode
    
    # Create a list of indices and shuffle them
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    idx = indices[:num_samples]

    # Create a DataLoader with SubsetRandomSampler
    sampler = SubsetRandomSampler(idx)
    testloader_random_samples = DataLoader(dataset, batch_size=64, sampler=sampler)

    correct_count, all_count = 0, 0
    for images, labels in testloader_random_samples:
        with torch.no_grad():
            outputs = model(images.view(images.shape[0], -1).to(device))
        
        _, predicted = torch.max(outputs, 1)
        correct_count += (predicted == labels.to(device)).sum().item()
        all_count += labels.size(0)
    
    print("Number of Samples tested =", all_count)
    print("\nModel Accuracy =", (correct_count / all_count))

# Assuming testset is your test dataset
# Call the function
infer_and_compute_accuracy_random_samples(model, testset, num_samples=1000)


# %%
model.layers[1](torch.randn(1, 28*28).to(device)).shape


# %%
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
model = DynamicMLP(num_layers=args.num_layers).to(device)
@torch.no_grad()
def mlp_sequential(model, dataloader, dev):
    print('Starting ...')

    # use_cache = model.config.use_cache
    # model.config.use_cache = False
    # layers = model.transformer.h
    layers = model.layers
    print("layers: ", layers)
    # model.transformer.word_embeddings = model.transformer.word_embeddings.to(dev)
    # model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(dev)
    layers[0] = layers[0].to(dev)
    
    print("layers[0]: ", layers[0])
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, args.batch_size, 28*28), dtype=dtype, device=dev
    )
    # cache = {'i': 0, 'attention_mask': None, 'alibi': None}
    cache = {'i': 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            # print("inps cache: ", inps[cache['i']])
            inps[cache['i']] = inp
            cache['i'] += 1
            # cache['attention_mask'] = kwargs['attention_mask']
            # cache['alibi'] = kwargs['alibi']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i, batch in enumerate(dataloader):
        try:
            print(i)
            print(batch[0].shape)
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    # model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
    # model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    # attention_mask = cache['attention_mask']
    # alibi = cache['alibi']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        gpts = {}
        for name in subset:
            if (not (args.minlayer <= i < args.maxlayer and args.prune_only in name)) == (not args.invert):
                continue
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]
            outs[j] = layer(inps[j].unsqueeze(0))[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('pruning ...')
            gpts[name].fasterprune(
                args.sparsity, prunen=args.prunen, prunem=args.prunem, percdamp=args.percdamp
            )
        for j in range(args.nsamples):
            # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]
            outs[j] = layer(inps[j].unsqueeze(0))[0]

        layers[i] = layer.cpu()
        del gpts 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    # model.config.use_cache = use_cache

# %%
mlp_sequential(model, testloader, device)


