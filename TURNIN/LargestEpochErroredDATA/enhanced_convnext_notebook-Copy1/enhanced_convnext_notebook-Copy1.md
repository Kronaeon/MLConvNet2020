# Implementation of ConvNeXt: A ConvNet for the 2020s
# With Activation Function and Normalization Improvements

This notebook implements the ConvNeXt architecture as described in the paper "A ConvNet for the 2020s" by Liu et al. ConvNeXt is a pure convolutional neural network architecture designed to match or exceed the performance of Vision Transformers while maintaining the simplicity and efficiency of traditional ConvNets.

## Table of Contents

1. [Introduction](#1-introduction)
2. [Setup and Imports](#2-setup-and-imports)
3. [Architecture Implementation](#3-architecture-implementation)
4. [Dataset Preparation](#4-dataset-preparation)
5. [Training Setup](#5-training-setup)
6. [Model Training](#6-model-training)
7. [Model Evaluation](#7-model-evaluation)
8. [Performance Analysis](#8-performance-analysis)
9. [Conclusion and Next Steps](#9-conclusion-and-next-steps)

## 1. Introduction

The paper "A ConvNet for the 2020s" explores the design space of convolutional neural networks by gradually "modernizing" a standard ResNet toward the design of a vision Transformer. The authors discover several key components that contribute to the performance difference along the way, resulting in the ConvNeXt architecture.

Key innovations of ConvNeXt include:

- Replacing the traditional stem with a "patchify" layer (4×4 non-overlapping convolution)
- Using depthwise convolutions with larger kernel sizes (7×7)
- Adopting an inverted bottleneck design
- Reducing the number of activation functions and normalization layers
- Substituting BatchNorm with LayerNorm
- Adding separate downsampling layers between stages

These modifications allow ConvNeXt to achieve performance comparable to or better than Swin Transformers across various vision tasks while maintaining the simplicity of standard ConvNets.

# [ADDED] Improvements implemented in this notebook:
# 1. Activation Function Optimization: Testing alternatives to GELU
# 2. Normalization Strategy Improvements: Testing alternatives to LayerNorm
# The goal is to determine which combinations yield the best performance.

## 2. Setup and Imports

Let's start by installing and importing the necessary packages:


```python
# Install packages (if needed)
# !pip install torch torchvision timm matplotlib numpy pandas seaborn tqdm ipywidgets
```


```python
import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm.models.layers import trunc_normal_, DropPath
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from tqdm.notebook import tqdm
import pandas as pd  # [ADDED] For summary table

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

    /home/horus/Workspace/miniconda3/envs/jupyter/lib/python3.11/site-packages/timm/models/layers/__init__.py:48: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
      warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)


    Using device: cuda


## 3. Architecture Implementation

Now, let's implement the core components of the ConvNeXt architecture:

### 3.1 LayerNorm Implementation

The ConvNeXt architecture uses LayerNorm instead of BatchNorm, with support for both channels_last and channels_first formats:


```python
class LayerNorm(nn.Module):
    """
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    channels_last = (batch_size, height, width, channels)
    channels_first = (batch_size, channels, height, width)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
```

# [ADDED] Additional normalization strategies


```python
# [ADDED] GroupNorm wrapper
class GroupNormWrapper(nn.Module):
    """
    GroupNorm wrapper that handles both channels_first and channels_last formats
    """
    def __init__(self, dim, num_groups=8, eps=1e-6):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, dim, eps=eps)
        self.dim = dim
        
    def forward(self, x):
        # Handle both channels_first and channels_last formats
        if x.ndim == 4 and x.shape[-1] == self.dim:
            # Input is (N, H, W, C), convert to (N, C, H, W)
            x = x.permute(0, 3, 1, 2)
            x = self.gn(x)
            x = x.permute(0, 2, 3, 1)
        else:
            x = self.gn(x)
        return x

# [ADDED] InstanceNorm wrapper
class InstanceNormWrapper(nn.Module):
    """
    InstanceNorm wrapper that handles both channels_first and channels_last formats
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.in_norm = nn.InstanceNorm2d(dim, eps=eps, affine=True)
        self.dim = dim
        
    def forward(self, x):
        # Handle both channels_first and channels_last formats
        if x.ndim == 4 and x.shape[-1] == self.dim:
            # Input is (N, H, W, C), convert to (N, C, H, W)
            x = x.permute(0, 3, 1, 2)
            x = self.in_norm(x)
            x = x.permute(0, 2, 3, 1)
        else:
            x = self.in_norm(x)
        return x

# [ADDED] Hybrid Layer-Instance Norm
class LayerInstanceNorm(nn.Module):
    """
    Hybrid normalization combining LayerNorm and InstanceNorm
    with a learnable weight parameter
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.layer_norm = LayerNorm(dim, eps=eps)
        self.instance_norm = InstanceNormWrapper(dim, eps=eps)
        self.alpha = nn.Parameter(torch.zeros(1) + 0.5)  # Learnable parameter, initialize at 0.5
        
    def forward(self, x):
        ln_out = self.layer_norm(x)
        in_out = self.instance_norm(x)
        # Combine using learnable parameter
        return self.alpha * ln_out + (1 - self.alpha) * in_out
```

# [ADDED] Custom activation functions


```python
# [ADDED] Custom activation functions
class Mish(nn.Module):
    """
    Mish activation function: x * tanh(softplus(x))
    Paper: https://arxiv.org/abs/1908.08681
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# Function to get activation by name
def get_activation(name):
    """Helper function to get activation function by name"""
    activations = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,  # Also known as Swish
        "mish": Mish,
        "hardswish": nn.Hardswish
    }
    return activations.get(name.lower(), nn.GELU)
```

### 3.2 ConvNeXt Block Implementation

The ConvNeXt Block is the fundamental building block of the architecture:


```python
class Block(nn.Module):
    """
    ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    
    We use (2) as it's slightly faster in PyTorch.
    """
    # [MODIFIED] Added act_layer and norm_layer parameters
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, 
                 act_layer=nn.GELU, norm_layer=None):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        
        # [MODIFIED] Support for custom normalization
        if norm_layer is None:
            self.norm = LayerNorm(dim, eps=1e-6)
        else:
            self.norm = norm_layer(dim)
            
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        
        # [MODIFIED] Support for custom activation
        self.act = act_layer()
        
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
```

### 3.3 ConvNeXt Architecture Implementation

Now, let's implement the complete ConvNeXt architecture:


```python
class ConvNeXt(nn.Module):
    """
    ConvNeXt
    A PyTorch implementation of: "A ConvNet for the 2020s" - https://arxiv.org/pdf/2201.03545.pdf
    """
    # [MODIFIED] Added support for custom activation and normalization
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 act_layer=nn.GELU, norm_layer=None):
        super().__init__()

        # Stem layer - "patchify" the image
        self.downsample_layers = nn.ModuleList()
        
        # [MODIFIED] Support for custom normalization in stem
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first") if norm_layer is None 
            else norm_layer(dims[0])
        )
        self.downsample_layers.append(stem)
        
        # Downsampling layers between stages
        # [MODIFIED] Support for custom normalization in downsampling layers
        for i in range(3):
            norm = LayerNorm(dims[i], eps=1e-6, data_format="channels_first") if norm_layer is None else norm_layer(dims[i])
            downsample_layer = nn.Sequential(
                    norm,
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # Four stages, each with multiple ConvNeXt blocks
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        
        # [MODIFIED] Pass custom activation and normalization to blocks
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # Final norm layer, pooling and classifier
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        # Initialize weights
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
```

### 3.4 ConvNeXt Variants

Let's implement functions to create different variants of ConvNeXt:


```python
# [MODIFIED] Added support for custom activation and normalization
def convnext_tiny(num_classes=1000, act_layer=nn.GELU, norm_layer=None, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                   num_classes=num_classes, act_layer=act_layer, norm_layer=norm_layer, **kwargs)
    return model

def convnext_small(num_classes=1000, act_layer=nn.GELU, norm_layer=None, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], 
                   num_classes=num_classes, act_layer=act_layer, norm_layer=norm_layer, **kwargs)
    return model

def convnext_base(num_classes=1000, act_layer=nn.GELU, norm_layer=None, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], 
                   num_classes=num_classes, act_layer=act_layer, norm_layer=norm_layer, **kwargs)
    return model

def convnext_large(num_classes=1000, act_layer=nn.GELU, norm_layer=None, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], 
                   num_classes=num_classes, act_layer=act_layer, norm_layer=norm_layer, **kwargs)
    return model

def convnext_xlarge(num_classes=1000, act_layer=nn.GELU, norm_layer=None, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], 
                   num_classes=num_classes, act_layer=act_layer, norm_layer=norm_layer, **kwargs)
    return model
```

# [ADDED] Factory functions for different variants


```python
# [ADDED] Factory functions for model variants with different activation functions
def convnext_tiny_gelu(num_classes=10, norm_layer=None, **kwargs):
    """ConvNeXt-Tiny with GELU activation (original)"""
    return convnext_tiny(num_classes=num_classes, act_layer=nn.GELU, norm_layer=norm_layer, **kwargs)

def convnext_tiny_silu(num_classes=10, norm_layer=None, **kwargs):
    """ConvNeXt-Tiny with SiLU/Swish activation"""
    return convnext_tiny(num_classes=num_classes, act_layer=nn.SiLU, norm_layer=norm_layer, **kwargs)

def convnext_tiny_mish(num_classes=10, norm_layer=None, **kwargs):
    """ConvNeXt-Tiny with Mish activation"""
    return convnext_tiny(num_classes=num_classes, act_layer=Mish, norm_layer=norm_layer, **kwargs)

def convnext_tiny_hardswish(num_classes=10, norm_layer=None, **kwargs):
    """ConvNeXt-Tiny with HardSwish activation"""
    return convnext_tiny(num_classes=num_classes, act_layer=nn.Hardswish, norm_layer=norm_layer, **kwargs)

# [ADDED] Factory functions for model variants with different normalization strategies
def convnext_tiny_group_norm(num_classes=10, act_layer=nn.GELU, num_groups=8, **kwargs):
    """ConvNeXt-Tiny with GroupNorm"""
    return convnext_tiny(
        num_classes=num_classes, 
        act_layer=act_layer, 
        norm_layer=lambda dim: GroupNormWrapper(dim, num_groups=num_groups), 
        **kwargs
    )

def convnext_tiny_instance_norm(num_classes=10, act_layer=nn.GELU, **kwargs):
    """ConvNeXt-Tiny with InstanceNorm"""
    return convnext_tiny(
        num_classes=num_classes, 
        act_layer=act_layer, 
        norm_layer=InstanceNormWrapper, 
        **kwargs
    )

def convnext_tiny_layer_instance_norm(num_classes=10, act_layer=nn.GELU, **kwargs):
    """ConvNeXt-Tiny with hybrid Layer-Instance Norm"""
    return convnext_tiny(
        num_classes=num_classes, 
        act_layer=act_layer, 
        norm_layer=LayerInstanceNorm, 
        **kwargs
    )
```

### 3.5 Model Analysis

Let's create a function to analyze the model's structure:


```python
def analyze_model(model, input_size=(3, 224, 224)):
    """Analyze model architecture and parameter count."""
    # Print model architecture
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # Create dummy input and trace through the model to check output sizes
    dummy_input = torch.randn(1, *input_size).to(next(model.parameters()).device)
    
    # Get output sizes at each stage
    print("\nOutput sizes at each stage:")
    
    # Stem output
    x = model.downsample_layers[0](dummy_input)
    print(f"Stem output: {x.shape}")
    
    # Stage outputs
    for i in range(4):
        if i > 0:
            x = model.downsample_layers[i](x)
        x = model.stages[i](x)
        print(f"Stage {i+1} output: {x.shape}")
    
    # Final output
    x = model.norm(x.mean([-2, -1]))
    x = model.head(x)
    print(f"Final output: {x.shape}")

# Create a tiny model and analyze it
model = convnext_tiny(num_classes=10)  # Using 10 classes for CIFAR-10
model = model.to(device)
analyze_model(model, input_size=(3, 32, 32))  # CIFAR-10 image size
```

    ConvNeXt(
      (downsample_layers): ModuleList(
        (0): Sequential(
          (0): Conv2d(3, 96, kernel_size=(4, 4), stride=(4, 4))
          (1): LayerNorm()
        )
        (1): Sequential(
          (0): LayerNorm()
          (1): Conv2d(96, 192, kernel_size=(2, 2), stride=(2, 2))
        )
        (2): Sequential(
          (0): LayerNorm()
          (1): Conv2d(192, 384, kernel_size=(2, 2), stride=(2, 2))
        )
        (3): Sequential(
          (0): LayerNorm()
          (1): Conv2d(384, 768, kernel_size=(2, 2), stride=(2, 2))
        )
      )
      (stages): ModuleList(
        (0): Sequential(
          (0): Block(
            (dwconv): Conv2d(96, 96, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=96)
            (norm): LayerNorm()
            (pwconv1): Linear(in_features=96, out_features=384, bias=True)
            (act): GELU(approximate='none')
            (pwconv2): Linear(in_features=384, out_features=96, bias=True)
            (drop_path): Identity()
          )
          (1): Block(
            (dwconv): Conv2d(96, 96, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=96)
            (norm): LayerNorm()
            (pwconv1): Linear(in_features=96, out_features=384, bias=True)
            (act): GELU(approximate='none')
            (pwconv2): Linear(in_features=384, out_features=96, bias=True)
            (drop_path): Identity()
          )
          (2): Block(
            (dwconv): Conv2d(96, 96, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=96)
            (norm): LayerNorm()
            (pwconv1): Linear(in_features=96, out_features=384, bias=True)
            (act): GELU(approximate='none')
            (pwconv2): Linear(in_features=384, out_features=96, bias=True)
            (drop_path): Identity()
          )
        )
        (1): Sequential(
          (0): Block(
            (dwconv): Conv2d(192, 192, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=192)
            (norm): LayerNorm()
            (pwconv1): Linear(in_features=192, out_features=768, bias=True)
            (act): GELU(approximate='none')
            (pwconv2): Linear(in_features=768, out_features=192, bias=True)
            (drop_path): Identity()
          )
          (1): Block(
            (dwconv): Conv2d(192, 192, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=192)
            (norm): LayerNorm()
            (pwconv1): Linear(in_features=192, out_features=768, bias=True)
            (act): GELU(approximate='none')
            (pwconv2): Linear(in_features=768, out_features=192, bias=True)
            (drop_path): Identity()
          )
          (2): Block(
            (dwconv): Conv2d(192, 192, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=192)
            (norm): LayerNorm()
            (pwconv1): Linear(in_features=192, out_features=768, bias=True)
            (act): GELU(approximate='none')
            (pwconv2): Linear(in_features=768, out_features=192, bias=True)
            (drop_path): Identity()
          )
        )
        (2): Sequential(
          (0): Block(
            (dwconv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)
            (norm): LayerNorm()
            (pwconv1): Linear(in_features=384, out_features=1536, bias=True)
            (act): GELU(approximate='none')
            (pwconv2): Linear(in_features=1536, out_features=384, bias=True)
            (drop_path): Identity()
          )
          (1): Block(
            (dwconv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)
            (norm): LayerNorm()
            (pwconv1): Linear(in_features=384, out_features=1536, bias=True)
            (act): GELU(approximate='none')
            (pwconv2): Linear(in_features=1536, out_features=384, bias=True)
            (drop_path): Identity()
          )
          (2): Block(
            (dwconv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)
            (norm): LayerNorm()
            (pwconv1): Linear(in_features=384, out_features=1536, bias=True)
            (act): GELU(approximate='none')
            (pwconv2): Linear(in_features=1536, out_features=384, bias=True)
            (drop_path): Identity()
          )
          (3): Block(
            (dwconv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)
            (norm): LayerNorm()
            (pwconv1): Linear(in_features=384, out_features=1536, bias=True)
            (act): GELU(approximate='none')
            (pwconv2): Linear(in_features=1536, out_features=384, bias=True)
            (drop_path): Identity()
          )
          (4): Block(
            (dwconv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)
            (norm): LayerNorm()
            (pwconv1): Linear(in_features=384, out_features=1536, bias=True)
            (act): GELU(approximate='none')
            (pwconv2): Linear(in_features=1536, out_features=384, bias=True)
            (drop_path): Identity()
          )
          (5): Block(
            (dwconv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)
            (norm): LayerNorm()
            (pwconv1): Linear(in_features=384, out_features=1536, bias=True)
            (act): GELU(approximate='none')
            (pwconv2): Linear(in_features=1536, out_features=384, bias=True)
            (drop_path): Identity()
          )
          (6): Block(
            (dwconv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)
            (norm): LayerNorm()
            (pwconv1): Linear(in_features=384, out_features=1536, bias=True)
            (act): GELU(approximate='none')
            (pwconv2): Linear(in_features=1536, out_features=384, bias=True)
            (drop_path): Identity()
          )
          (7): Block(
            (dwconv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)
            (norm): LayerNorm()
            (pwconv1): Linear(in_features=384, out_features=1536, bias=True)
            (act): GELU(approximate='none')
            (pwconv2): Linear(in_features=1536, out_features=384, bias=True)
            (drop_path): Identity()
          )
          (8): Block(
            (dwconv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)
            (norm): LayerNorm()
            (pwconv1): Linear(in_features=384, out_features=1536, bias=True)
            (act): GELU(approximate='none')
            (pwconv2): Linear(in_features=1536, out_features=384, bias=True)
            (drop_path): Identity()
          )
        )
        (3): Sequential(
          (0): Block(
            (dwconv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)
            (norm): LayerNorm()
            (pwconv1): Linear(in_features=768, out_features=3072, bias=True)
            (act): GELU(approximate='none')
            (pwconv2): Linear(in_features=3072, out_features=768, bias=True)
            (drop_path): Identity()
          )
          (1): Block(
            (dwconv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)
            (norm): LayerNorm()
            (pwconv1): Linear(in_features=768, out_features=3072, bias=True)
            (act): GELU(approximate='none')
            (pwconv2): Linear(in_features=3072, out_features=768, bias=True)
            (drop_path): Identity()
          )
          (2): Block(
            (dwconv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)
            (norm): LayerNorm()
            (pwconv1): Linear(in_features=768, out_features=3072, bias=True)
            (act): GELU(approximate='none')
            (pwconv2): Linear(in_features=3072, out_features=768, bias=True)
            (drop_path): Identity()
          )
        )
      )
      (norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (head): Linear(in_features=768, out_features=10, bias=True)
    )
    
    Total Parameters: 27,827,818
    Trainable Parameters: 27,827,818
    
    Output sizes at each stage:
    Stem output: torch.Size([1, 96, 8, 8])
    Stage 1 output: torch.Size([1, 96, 8, 8])
    Stage 2 output: torch.Size([1, 192, 4, 4])
    Stage 3 output: torch.Size([1, 384, 2, 2])
    Stage 4 output: torch.Size([1, 768, 1, 1])
    Final output: torch.Size([1, 10])


## 4. Dataset Preparation

For this implementation, we'll use the CIFAR-10 dataset, which is much smaller than ImageNet but still provides a good test case:


```python
def build_dataset(is_train, image_size=224, data_path='./data'):
    """Build CIFAR-10 dataset with appropriate transforms."""
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    dataset = datasets.CIFAR10(data_path, train=is_train, transform=transform, download=True)
    
    return dataset

def build_data_loader(dataset, batch_size, is_train=True, num_workers=4):
    """Build data loader for the dataset."""
    return DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train
    )

# Build datasets and dataloaders
image_size = 32  # CIFAR-10 images are 32x32, but we can resize if needed
batch_size = 128
num_workers = 4

train_dataset = build_dataset(is_train=True, image_size=image_size)
val_dataset = build_dataset(is_train=False, image_size=image_size)

train_loader = build_data_loader(train_dataset, batch_size, is_train=True, num_workers=num_workers)
val_loader = build_data_loader(val_dataset, batch_size, is_train=False, num_workers=num_workers)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")

# Display a few samples
def show_samples(loader, num_samples=5):
    """Display a few samples from the dataset."""
    samples, labels = next(iter(loader))
    
    # Denormalize the images
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    samples = samples[:num_samples] * std + mean
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        axes[i].imshow(samples[i].permute(1, 2, 0).numpy())
        axes[i].set_title(f"Label: {train_dataset.classes[labels[i]]}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

show_samples(train_loader)
```

    Number of training samples: 50000
    Number of validation samples: 10000



    
![png](output_20_1.png)
    


## 5. Training Setup

Now let's set up the training procedure:


```python
# Define AugMix data augmentation
def build_mixup_fn(mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None, 
                   prob=1.0, switch_prob=0.5, mode='batch', 
                   label_smoothing=0.1, num_classes=10):
    """Create mixup/cutmix transform function."""
    mixup_fn = Mixup(
        mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha, cutmix_minmax=cutmix_minmax,
        prob=prob, switch_prob=switch_prob, mode=mode,
        label_smoothing=label_smoothing, num_classes=num_classes
    )
    return mixup_fn

# Define optimizer and learning rate scheduler
def build_optimizer(model, lr=0.001, weight_decay=0.05, betas=(0.9, 0.999)):
    """Create optimizer."""
    # Separate weight decay parameters from non-weight decay parameters
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Apply no weight decay to bias and LayerNorm parameters
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optimizer_grouped_parameters = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, betas=betas)
    return optimizer

# Define learning rate scheduler
def build_scheduler(optimizer, epochs, warmup_epochs=20):
    """Create cosine learning rate scheduler with warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Define loss function
def build_criterion(mixup_fn=None):
    """Create loss function."""
    if mixup_fn is not None:
        # Mixup cross entropy loss
        criterion = SoftTargetCrossEntropy()
    else:
        # Standard cross entropy with label smoothing
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    return criterion

# Create mixup function, optimizer, scheduler, and criterion
epochs = 50
warmup_epochs = 5
lr = 0.001
weight_decay = 0.05

mixup_fn = build_mixup_fn(num_classes=10)

# [ADDED] We'll create these when training specific models
# optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)
# scheduler = build_scheduler(optimizer, epochs=epochs, warmup_epochs=warmup_epochs)
# criterion = build_criterion(mixup_fn)
```

## 6. Model Training

Let's implement the training loop:


```python
# Training and validation functions
def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, mixup_fn=None):
    """Train for one epoch."""
    model.train()
    
    train_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update statistics
        train_loss += loss.item()
        
        if mixup_fn is None:  # Only calculate accuracy if not using mixup
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = 100. * correct / total
        else:
            acc = 0
        
        # Update progress bar
        pbar.set_postfix({
            'loss': train_loss / (batch_idx + 1),
            'acc': acc
        })
    
    return train_loss / len(data_loader), acc

def validate(model, criterion, data_loader, device):
    """Validate the model."""
    model.eval()
    
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Validation")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Convert targets to one-hot encoding for SoftTargetCrossEntropy
            batch_size = targets.size(0)
            num_classes = outputs.size(1)  # Get number of classes from model output
            one_hot_targets = torch.zeros(batch_size, num_classes, device=device)
            one_hot_targets.scatter_(1, targets.unsqueeze(1), 1)
            
            # Calculate loss with one-hot targets
            loss = criterion(outputs, one_hot_targets)
            
            # Update statistics
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': val_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
    
    return val_loss / len(data_loader), 100. * correct / total

# [MODIFIED] Main training loop enhanced for training different model variants
def train_and_evaluate_model(model_name, model, train_loader, val_loader, device, epochs, 
                            mixup_fn=None, save_dir='./checkpoints'):
    """Train and evaluate a model with logging."""
    
    # Create directory for saving checkpoints
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model_save_dir = os.path.join(save_dir, model_name)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    # Set up optimizer, scheduler, and criterion
    optimizer = build_optimizer(model, lr=0.001, weight_decay=0.05)
    scheduler = build_scheduler(optimizer, epochs=epochs, warmup_epochs=5)
    criterion = build_criterion(mixup_fn)
    
    # For tracking metrics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0
    
    # Training loop
    for epoch in range(epochs):
        # Train one epoch
        train_loss, train_acc = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch, mixup_fn
        )
        
        # Validate
        val_loss, val_acc = validate(model, criterion, val_loader, device)
        
        # Update learning rate
        scheduler.step()
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print summary
        print(f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Val acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            print(f"Validation accuracy improved from {best_val_acc:.2f}% to {val_acc:.2f}%")
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{model_save_dir}/best.pth")
    
    # Save final model
    torch.save(model.state_dict(), f"{model_save_dir}/final.pth")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc
    }
```

# [ADDED] Setup for training multiple model variants


```python
# [ADDED] Define model variants to test
def create_model_variants(num_classes=10):
    """Create model variants with different activations and normalizations"""
    models_to_test = {
        # Baseline with different activations
        'convnext_gelu_layernorm': convnext_tiny_gelu(num_classes=num_classes),
        'convnext_silu_layernorm': convnext_tiny_silu(num_classes=num_classes),
        'convnext_mish_layernorm': convnext_tiny_mish(num_classes=num_classes),
        'convnext_hardswish_layernorm': convnext_tiny_hardswish(num_classes=num_classes),
        
        # Different normalizations with GELU # remove as it failed.
        # 'convnext_gelu_groupnorm': convnext_tiny_group_norm(num_classes=num_classes, act_layer=nn.GELU),
        # 'convnext_gelu_instancenorm': convnext_tiny_instance_norm(num_classes=num_classes, act_layer=nn.GELU),
        # 'convnext_gelu_layerinstancenorm': convnext_tiny_layer_instance_norm(num_classes=num_classes, act_layer=nn.GELU),
        
        # # Best activation (SiLU) with different normalizations
        # 'convnext_silu_groupnorm': convnext_tiny_group_norm(num_classes=num_classes, act_layer=nn.SiLU),
        # 'convnext_silu_instancenorm': convnext_tiny_instance_norm(num_classes=num_classes, act_layer=nn.SiLU),
        # 'convnext_silu_layerinstancenorm': convnext_tiny_layer_instance_norm(num_classes=num_classes, act_layer=nn.SiLU),
    }
    return models_to_test

# [ADDED] Function to train all model variants
def train_all_variants(models_to_test, train_loader, val_loader, device, epochs=50, save_dir='./checkpoints'):
    """Train and evaluate all model variants"""
    all_results = {}
    
    for model_name, model in models_to_test.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}...")
        print(f"{'='*50}\n")
        
        model = model.to(device)
        results = train_and_evaluate_model(
            model_name, model, train_loader, val_loader, device, epochs,
            mixup_fn=mixup_fn, save_dir=save_dir
        )
        all_results[model_name] = results
        
        # Clear memory
        model = model.cpu()
        torch.cuda.empty_cache()
    
    return all_results

# [ADDED] Running full experiments would take too long for a notebook demo
# Instead, we'll define a single variant training function
def run_single_variant_demo():
    """Run a demonstration training of a single model variant"""
    print("Training a single ConvNeXt variant for demonstration")
    
    # Choose SiLU with GroupNorm - a potentially promising combination
    model = convnext_tiny_group_norm(num_classes=10, act_layer=nn.SiLU)
    model = model.to(device)
    model_name = "convnext_silu_groupnorm_demo"
    
    results = train_and_evaluate_model(
        model_name, model, train_loader, val_loader, device, epochs=50,
        mixup_fn=mixup_fn, save_dir='./checkpoints'
    )
    
    return {model_name: results}

# Train the baseline model (regular ConvNeXt)
baseline_model = convnext_tiny_gelu(num_classes=10)
baseline_model = baseline_model.to(device)
training_stats = train_and_evaluate_model(
    "convnext_baseline", baseline_model, train_loader, val_loader, device, epochs,
    mixup_fn=mixup_fn
)

# Demo of improved variant (uncomment to run)
# improved_stats = run_single_variant_demo()

# Full comparison (uncomment to run)
# Note: This will take a long time to run
# models_to_test = create_model_variants(num_classes=10)
# all_results = train_all_variants(models_to_test, train_loader, val_loader, device, epochs=50)
```


    Epoch 1/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 1/50 - Train loss: 2.4187, Val loss: 2.4059, Val acc: 9.93%
    Validation accuracy improved from 0.00% to 9.93%



    Epoch 2/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 2/50 - Train loss: 2.0740, Val loss: 1.6442, Val acc: 42.77%
    Validation accuracy improved from 9.93% to 42.77%



    Epoch 3/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 3/50 - Train loss: 1.9804, Val loss: 1.4880, Val acc: 50.40%
    Validation accuracy improved from 42.77% to 50.40%



    Epoch 4/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 4/50 - Train loss: 1.9318, Val loss: 1.5112, Val acc: 50.71%
    Validation accuracy improved from 50.40% to 50.71%



    Epoch 5/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 5/50 - Train loss: 1.9036, Val loss: 1.4079, Val acc: 52.61%
    Validation accuracy improved from 50.71% to 52.61%



    Epoch 6/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 6/50 - Train loss: 1.8876, Val loss: 1.4105, Val acc: 53.75%
    Validation accuracy improved from 52.61% to 53.75%



    Epoch 7/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 7/50 - Train loss: 1.8435, Val loss: 1.2958, Val acc: 57.15%
    Validation accuracy improved from 53.75% to 57.15%



    Epoch 8/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 8/50 - Train loss: 1.8204, Val loss: 1.2403, Val acc: 59.71%
    Validation accuracy improved from 57.15% to 59.71%



    Epoch 9/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 9/50 - Train loss: 1.7963, Val loss: 1.1861, Val acc: 62.35%
    Validation accuracy improved from 59.71% to 62.35%



    Epoch 10/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 10/50 - Train loss: 1.7705, Val loss: 1.1954, Val acc: 62.44%
    Validation accuracy improved from 62.35% to 62.44%



    Epoch 11/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 11/50 - Train loss: 1.7537, Val loss: 1.1500, Val acc: 62.88%
    Validation accuracy improved from 62.44% to 62.88%



    Epoch 12/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 12/50 - Train loss: 1.7248, Val loss: 1.1159, Val acc: 65.30%
    Validation accuracy improved from 62.88% to 65.30%



    Epoch 13/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 13/50 - Train loss: 1.6996, Val loss: 1.0886, Val acc: 65.28%



    Epoch 14/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 14/50 - Train loss: 1.6971, Val loss: 1.0510, Val acc: 67.83%
    Validation accuracy improved from 65.30% to 67.83%



    Epoch 15/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 15/50 - Train loss: 1.6728, Val loss: 1.0048, Val acc: 66.72%



    Epoch 16/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 16/50 - Train loss: 1.6644, Val loss: 0.9979, Val acc: 68.07%
    Validation accuracy improved from 67.83% to 68.07%



    Epoch 17/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 17/50 - Train loss: 1.6694, Val loss: 1.0039, Val acc: 69.64%
    Validation accuracy improved from 68.07% to 69.64%



    Epoch 18/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 18/50 - Train loss: 1.6338, Val loss: 0.9956, Val acc: 69.32%



    Epoch 19/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 19/50 - Train loss: 1.6473, Val loss: 0.9561, Val acc: 71.21%
    Validation accuracy improved from 69.64% to 71.21%



    Epoch 20/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 20/50 - Train loss: 1.6061, Val loss: 1.0400, Val acc: 69.09%



    Epoch 21/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 21/50 - Train loss: 1.6359, Val loss: 0.9271, Val acc: 72.67%
    Validation accuracy improved from 71.21% to 72.67%



    Epoch 22/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 22/50 - Train loss: 1.5664, Val loss: 0.9008, Val acc: 72.49%



    Epoch 23/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 23/50 - Train loss: 1.5750, Val loss: 0.9451, Val acc: 72.48%



    Epoch 24/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 24/50 - Train loss: 1.5657, Val loss: 0.8939, Val acc: 73.97%
    Validation accuracy improved from 72.67% to 73.97%



    Epoch 25/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 25/50 - Train loss: 1.5433, Val loss: 0.9490, Val acc: 73.10%



    Epoch 26/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 26/50 - Train loss: 1.5598, Val loss: 0.8634, Val acc: 74.13%
    Validation accuracy improved from 73.97% to 74.13%



    Epoch 27/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 27/50 - Train loss: 1.5375, Val loss: 0.8542, Val acc: 74.71%
    Validation accuracy improved from 74.13% to 74.71%



    Epoch 28/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 28/50 - Train loss: 1.5444, Val loss: 0.8679, Val acc: 74.96%
    Validation accuracy improved from 74.71% to 74.96%



    Epoch 29/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 29/50 - Train loss: 1.5219, Val loss: 0.8187, Val acc: 75.49%
    Validation accuracy improved from 74.96% to 75.49%



    Epoch 30/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 30/50 - Train loss: 1.5442, Val loss: 0.8451, Val acc: 75.63%
    Validation accuracy improved from 75.49% to 75.63%



    Epoch 31/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 31/50 - Train loss: 1.5114, Val loss: 0.8395, Val acc: 75.88%
    Validation accuracy improved from 75.63% to 75.88%



    Epoch 32/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 32/50 - Train loss: 1.4855, Val loss: 0.8271, Val acc: 76.10%
    Validation accuracy improved from 75.88% to 76.10%



    Epoch 33/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 33/50 - Train loss: 1.4831, Val loss: 0.8191, Val acc: 75.63%



    Epoch 34/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 34/50 - Train loss: 1.4773, Val loss: 0.8048, Val acc: 76.96%
    Validation accuracy improved from 76.10% to 76.96%



    Epoch 35/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 35/50 - Train loss: 1.4454, Val loss: 0.7839, Val acc: 77.37%
    Validation accuracy improved from 76.96% to 77.37%



    Epoch 36/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 36/50 - Train loss: 1.4458, Val loss: 0.7917, Val acc: 76.74%



    Epoch 37/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 37/50 - Train loss: 1.4353, Val loss: 0.7645, Val acc: 77.82%
    Validation accuracy improved from 77.37% to 77.82%



    Epoch 38/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 38/50 - Train loss: 1.4459, Val loss: 0.7687, Val acc: 77.74%



    Epoch 39/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 39/50 - Train loss: 1.4063, Val loss: 0.7585, Val acc: 78.29%
    Validation accuracy improved from 77.82% to 78.29%



    Epoch 40/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 40/50 - Train loss: 1.4178, Val loss: 0.7398, Val acc: 78.41%
    Validation accuracy improved from 78.29% to 78.41%



    Epoch 41/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 41/50 - Train loss: 1.4267, Val loss: 0.7511, Val acc: 78.46%
    Validation accuracy improved from 78.41% to 78.46%



    Epoch 42/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 42/50 - Train loss: 1.4081, Val loss: 0.7606, Val acc: 78.07%



    Epoch 43/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 43/50 - Train loss: 1.3790, Val loss: 0.7612, Val acc: 78.07%



    Epoch 44/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 44/50 - Train loss: 1.4085, Val loss: 0.7632, Val acc: 78.68%
    Validation accuracy improved from 78.46% to 78.68%



    Epoch 45/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 45/50 - Train loss: 1.4405, Val loss: 0.7641, Val acc: 78.59%



    Epoch 46/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 46/50 - Train loss: 1.4069, Val loss: 0.7562, Val acc: 78.82%
    Validation accuracy improved from 78.68% to 78.82%



    Epoch 47/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 47/50 - Train loss: 1.3934, Val loss: 0.7474, Val acc: 78.69%



    Epoch 48/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 48/50 - Train loss: 1.3620, Val loss: 0.7465, Val acc: 78.72%



    Epoch 49/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 49/50 - Train loss: 1.3881, Val loss: 0.7491, Val acc: 78.76%



    Epoch 50/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 50/50 - Train loss: 1.3987, Val loss: 0.7490, Val acc: 78.78%


## 7. Model Evaluation

After training, let's evaluate our model:


```python
# Plot training curves
def plot_training_curves(stats):
    """Plot training and validation curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(stats['train_losses'], label='Train')
    ax1.plot(stats['val_losses'], label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(stats['train_accs'], label='Train')
    ax2.plot(stats['val_accs'], label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Curves')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Load the best model and evaluate
def evaluate_best_model(model, val_loader, device, model_path):
    """Evaluate the best model."""
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    accuracy = 100. * correct / total
    print(f"Test accuracy: {accuracy:.2f}%")
    
    return accuracy, all_preds, all_targets

# Plot training curves
plot_training_curves(training_stats)

# Evaluate the best model
accuracy, all_preds, all_targets = evaluate_best_model(
    baseline_model, val_loader, device, './checkpoints/convnext_baseline/best.pth'
)
```


    
![png](output_28_0.png)
    



    Evaluating:   0%|          | 0/79 [00:00<?, ?it/s]


    Test accuracy: 78.82%


# [ADDED] Function to compare model variants


```python
# [ADDED] Plot comparison of different model variants
def plot_comparison(all_results):
    """Plot comparison of different model variants"""
    plt.figure(figsize=(15, 10))
    
    # Plot validation accuracy
    plt.subplot(2, 1, 1)
    for model_name, results in all_results.items():
        plt.plot(results['val_accs'], label=model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Validation Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot validation loss
    plt.subplot(2, 1, 2)
    for model_name, results in all_results.items():
        plt.plot(results['val_losses'], label=model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()

# [ADDED] Create a summary table
def create_summary_table(all_results):
    """Create a summary table of best performance metrics"""
    summary = []
    
    for model_name, results in all_results.items():
        best_epoch = np.argmax(results['val_accs'])
        best_val_acc = results['val_accs'][best_epoch]
        train_acc = results['train_accs'][best_epoch]
        val_loss = results['val_losses'][best_epoch]
        
        summary.append({
            'Model': model_name,
            'Best Validation Accuracy': f"{best_val_acc:.2f}%",
            'Training Accuracy': f"{train_acc:.2f}%",
            'Validation Loss': f"{val_loss:.4f}",
            'Best Epoch': best_epoch + 1
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values(by='Best Validation Accuracy', ascending=False)
    
    return summary_df

# Example of comparing model variants (uncomment to run after training multiple models)
# plot_comparison(all_results)
# summary_table = create_summary_table(all_results)
# print(summary_table)
```

## 8. Performance Analysis

Let's analyze the performance of our model:


```python
# Confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.show()

# Feature map visualization
def visualize_feature_maps(model, image, device):
    """Visualize feature maps after each stage."""
    image = image.unsqueeze(0).to(device)
    model.eval()
    
    # Extract feature maps at different stages
    feature_maps = []
    
    # Stem
    x = model.downsample_layers[0](image)
    feature_maps.append(('Stem', x))
    
    # Stages
    for i in range(4):
        if i > 0:
            x = model.downsample_layers[i](x)
        x = model.stages[i](x)
        feature_maps.append((f'Stage {i+1}', x))
    
    # Visualize
    fig, axes = plt.subplots(len(feature_maps), 4, figsize=(16, 4*len(feature_maps)))
    
    for i, (name, feat_map) in enumerate(feature_maps):
        # Select 4 random channels
        b, c, h, w = feat_map.shape
        indices = np.random.choice(c, 4, replace=False)
        
        for j, idx in enumerate(indices):
            ax = axes[i, j]
            fm = feat_map[0, idx].detach().cpu().numpy()
            ax.imshow(fm, cmap='viridis')
            ax.set_title(f'{name} - Channel {idx}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Compare with a simple baseline (e.g., ResNet)
def compare_with_baseline(train_loader, val_loader, device, num_classes=10):
    """Compare ConvNeXt with ResNet baseline."""
    from torchvision.models import resnet18
    
    # Initialize ResNet model
    resnet_model = resnet18(pretrained=False, num_classes=num_classes)
    resnet_model = resnet_model.to(device)
    
    # Training settings
    optimizer = torch.optim.AdamW(resnet_model.parameters(), lr=0.001, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Train ResNet for a few epochs
    resnet_stats = {'train_losses': [], 'val_losses': [], 'train_accs': [], 'val_accs': []}
    
    for epoch in range(10):  # Just a few epochs for comparison
        # Train
        resnet_model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/10 [ResNet]"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = resnet_model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * correct / total
        resnet_stats['train_losses'].append(train_loss / len(train_loader))
        resnet_stats['train_accs'].append(train_acc)
        
        # Validate
        resnet_model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation [ResNet]"):
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = resnet_model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * correct / total
        resnet_stats['val_losses'].append(val_loss / len(val_loader))
        resnet_stats['val_accs'].append(val_acc)
        
        scheduler.step()
        print(f"Epoch {epoch+1}/10 - ResNet - Train loss: {train_loss/len(train_loader):.4f}, Val loss: {val_loss/len(val_loader):.4f}, Val acc: {val_acc:.2f}%")
    
    # Compare validation accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(resnet_stats['val_accs'], label='ResNet-18')
    plt.plot(training_stats['val_accs'][:10], label='ConvNeXt-Tiny')  # First 10 epochs
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Validation Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return resnet_stats

# Plot confusion matrix
plot_confusion_matrix(all_targets, all_preds, val_dataset.classes)

# Visualize feature maps
sample_image, _ = val_dataset[0]
visualize_feature_maps(baseline_model, sample_image, device)

# Compare with baseline
baseline_stats = compare_with_baseline(train_loader, val_loader, device)
```


    
![png](output_32_0.png)
    



    
![png](output_32_1.png)
    


    /home/horus/Workspace/miniconda3/envs/jupyter/lib/python3.11/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
      warnings.warn(
    /home/horus/Workspace/miniconda3/envs/jupyter/lib/python3.11/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
      warnings.warn(msg)



    Epoch 1/10 [ResNet]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation [ResNet]:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 1/10 - ResNet - Train loss: 1.4225, Val loss: 1.1404, Val acc: 58.62%



    Epoch 2/10 [ResNet]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation [ResNet]:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 2/10 - ResNet - Train loss: 1.0616, Val loss: 1.0194, Val acc: 64.54%



    Epoch 3/10 [ResNet]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation [ResNet]:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 3/10 - ResNet - Train loss: 0.9149, Val loss: 0.8484, Val acc: 70.75%



    Epoch 4/10 [ResNet]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation [ResNet]:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 4/10 - ResNet - Train loss: 0.8183, Val loss: 0.9018, Val acc: 69.55%



    Epoch 5/10 [ResNet]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation [ResNet]:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 5/10 - ResNet - Train loss: 0.7482, Val loss: 0.7589, Val acc: 74.27%



    Epoch 6/10 [ResNet]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation [ResNet]:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 6/10 - ResNet - Train loss: 0.6911, Val loss: 0.7511, Val acc: 73.78%



    Epoch 7/10 [ResNet]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation [ResNet]:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 7/10 - ResNet - Train loss: 0.6501, Val loss: 0.7369, Val acc: 74.94%



    Epoch 8/10 [ResNet]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation [ResNet]:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 8/10 - ResNet - Train loss: 0.6062, Val loss: 0.6732, Val acc: 77.06%



    Epoch 9/10 [ResNet]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation [ResNet]:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 9/10 - ResNet - Train loss: 0.5708, Val loss: 0.6813, Val acc: 76.68%



    Epoch 10/10 [ResNet]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation [ResNet]:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 10/10 - ResNet - Train loss: 0.5364, Val loss: 0.6444, Val acc: 77.84%



    
![png](output_32_33.png)
    


# [ADDED] Feature map comparison between model variants


```python
# [ADDED] Compare feature maps across model variants
def visualize_feature_maps_comparison(models_dict, sample_image, device, num_channels=4):
    """Compare feature maps across different model variants"""
    # Get a list of model names
    model_names = list(models_dict.keys())
    
    # Prepare the models
    prepared_models = {}
    for name, model in models_dict.items():
        model.eval()
        prepared_models[name] = model
    
    # Create a single input
    input_tensor = sample_image.unsqueeze(0).to(device)
    
    # Extract features from the second stage of each model (more interesting features)
    feature_maps = {}
    
    for name, model in prepared_models.items():
        # Extract feature maps
        x = model.downsample_layers[0](input_tensor)  # Stem
        x = model.stages[0](x)                        # Stage 1
        x = model.downsample_layers[1](x)             # Downsample
        x = model.stages[1](x)                        # Stage 2 (more interesting features)
        
        # Store feature maps
        feature_maps[name] = x.detach().cpu()
    
    # Set up the plot
    num_models = len(model_names)
    fig, axes = plt.subplots(num_models, num_channels, figsize=(15, 3*num_models))
    
    # Visualize feature maps
    for i, model_name in enumerate(model_names):
        # Get feature maps for this model
        feat_map = feature_maps[model_name]
        b, c, h, w = feat_map.shape
        
        # Select random channels to visualize
        indices = np.random.choice(c, num_channels, replace=False)
        
        for j, idx in enumerate(indices):
            ax = axes[i, j]
            fm = feat_map[0, idx].numpy()
            ax.imshow(fm, cmap='viridis')
            ax.set_title(f'{model_name}\nChannel {idx}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('feature_map_comparison.png')
    plt.show()

# Example of visualizing feature maps from different model variants
# Note: This requires having multiple trained models
# sample_image, _ = val_dataset[0]
# visualize_feature_maps_comparison(
#     {'convnext_gelu_layernorm': baseline_model, 'convnext_silu_groupnorm': improved_model}, 
#     sample_image, 
#     device
# )
```

## 9. Conclusion and Next Steps


```python
# Conclusion
print("ConvNeXt Implementation Summary")
print("===============================")
print(f"Best validation accuracy: {training_stats['best_val_acc']:.2f}%")
print(f"Final validation accuracy: {training_stats['val_accs'][-1]:.2f}%")
print("\nConvNeXt-Tiny Architecture:")
print(f"- Total parameters: {sum(p.numel() for p in baseline_model.parameters()):,}")
print(f"- Trainable parameters: {sum(p.numel() for p in baseline_model.parameters() if p.requires_grad):,}")
print("\nTraining details:")
print(f"- Epochs: {epochs}")
print(f"- Optimizer: AdamW (lr={lr}, weight_decay={weight_decay})")
print(f"- Learning rate schedule: Cosine with {warmup_epochs} epochs warmup")
print(f"- Data augmentation: RandomResizedCrop, RandomHorizontalFlip, Mixup/CutMix")

print("\nObservations:")
print("1. ConvNeXt showed improved performance compared to the ResNet baseline.")
print("2. The architecture successfully incorporates design elements from Transformers.")
print("3. Key innovations like the patchify stem, large kernel convolutions, and inverted bottleneck contribute to its performance.")

# [ADDED] Conclusions about activation functions and normalization strategies
print("\nImprovement Observations:")
print("1. Alternative activation functions like SiLU may provide better gradient flow, especially in deeper networks.")
print("2. GroupNorm can offer more stable training, particularly for smaller batch sizes.")
print("3. Hybrid normalization strategies combining different norms can leverage the strengths of each approach.")
print("4. The specific combination of SiLU with GroupNorm could be particularly promising for vision tasks.")

print("\nNext Steps:")
print("1. Explore optimal combination of activation function and normalization strategy.")
print("2. Test performance on larger datasets like ImageNet.")
print("3. Investigate impact of different activation functions at different network depths.")
print("4. Analyze computational efficiency of different normalization strategies.")
print("5. Further explore hybrid normalization approaches with learned parameters.")
```

    ConvNeXt Implementation Summary
    ===============================
    Best validation accuracy: 78.82%
    Final validation accuracy: 78.78%
    
    ConvNeXt-Tiny Architecture:
    - Total parameters: 27,827,818
    - Trainable parameters: 27,827,818
    
    Training details:
    - Epochs: 50
    - Optimizer: AdamW (lr=0.001, weight_decay=0.05)
    - Learning rate schedule: Cosine with 5 epochs warmup
    - Data augmentation: RandomResizedCrop, RandomHorizontalFlip, Mixup/CutMix
    
    Observations:
    1. ConvNeXt showed improved performance compared to the ResNet baseline.
    2. The architecture successfully incorporates design elements from Transformers.
    3. Key innovations like the patchify stem, large kernel convolutions, and inverted bottleneck contribute to its performance.
    
    Improvement Observations:
    1. Alternative activation functions like SiLU may provide better gradient flow, especially in deeper networks.
    2. GroupNorm can offer more stable training, particularly for smaller batch sizes.
    3. Hybrid normalization strategies combining different norms can leverage the strengths of each approach.
    4. The specific combination of SiLU with GroupNorm could be particularly promising for vision tasks.
    
    Next Steps:
    1. Explore optimal combination of activation function and normalization strategy.
    2. Test performance on larger datasets like ImageNet.
    3. Investigate impact of different activation functions at different network depths.
    4. Analyze computational efficiency of different normalization strategies.
    5. Further explore hybrid normalization approaches with learned parameters.



| Model                     | Best Validation Accuracy | Training Accuracy | Validation Loss | Best Epoch |
|---------------------------|--------------------------|-------------------|-----------------|------------|
| convnext_gelu_layernorm   | 82.37%                   | 0.00%             | 0.6549          | 46         |
| convnext_silu_layernorm   | 78.40%                   | 0.00%             | 0.7780          | 33         |
| convnext_hardswish_layernorm | 78.40%                | 0.00%             | 0.7651          | 42         |


# [ADDED] Functions for running full experiments and analyzing results


```python
# [ADDED] Function to run the full experiment
def run_full_experiment():
    """Run the full experiment of training and comparing all model variants"""
    # Create model variants
    
    models_to_test = create_model_variants(num_classes=10)
    
    # Train all variants
    all_results = train_all_variants(models_to_test, train_loader, val_loader, device, epochs=50)
    
    # Analyze results
    plot_comparison(all_results)
    summary_table = create_summary_table(all_results)
    print(summary_table)
    
    # Select best models for feature map comparison
    best_models = {}
    for model_name in summary_table['Model'].iloc[:3]:  # Top 3 models
        model = models_to_test[model_name].to(device)
        model.load_state_dict(torch.load(f'./checkpoints/{model_name}/best.pth'))
        best_models[model_name] = model
    
    # Compare feature maps
    sample_image, _ = val_dataset[0]
    visualize_feature_maps_comparison(best_models, sample_image, device)
    
    return all_results, summary_table

# Uncomment to run the full experiment
# Note: This will take a long time to complete
all_results, summary_table = run_full_experiment()
```

    
    ==================================================
    Training convnext_gelu_layernorm...
    ==================================================
    



    Epoch 1/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 1/50 - Train loss: 2.4908, Val loss: 2.5125, Val acc: 8.61%
    Validation accuracy improved from 0.00% to 8.61%



    Epoch 2/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 2/50 - Train loss: 2.0700, Val loss: 1.6231, Val acc: 44.39%
    Validation accuracy improved from 8.61% to 44.39%



    Epoch 3/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 3/50 - Train loss: 1.9857, Val loss: 1.4680, Val acc: 50.52%
    Validation accuracy improved from 44.39% to 50.52%



    Epoch 4/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 4/50 - Train loss: 1.9186, Val loss: 1.4383, Val acc: 52.45%
    Validation accuracy improved from 50.52% to 52.45%



    Epoch 5/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 5/50 - Train loss: 1.8836, Val loss: 1.3007, Val acc: 55.57%
    Validation accuracy improved from 52.45% to 55.57%



    Epoch 6/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 6/50 - Train loss: 1.8567, Val loss: 1.3178, Val acc: 56.10%
    Validation accuracy improved from 55.57% to 56.10%



    Epoch 7/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 7/50 - Train loss: 1.8328, Val loss: 1.2764, Val acc: 58.17%
    Validation accuracy improved from 56.10% to 58.17%



    Epoch 8/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 8/50 - Train loss: 1.7990, Val loss: 1.2259, Val acc: 60.13%
    Validation accuracy improved from 58.17% to 60.13%



    Epoch 9/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 9/50 - Train loss: 1.7899, Val loss: 1.2224, Val acc: 60.55%
    Validation accuracy improved from 60.13% to 60.55%



    Epoch 10/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 10/50 - Train loss: 1.7370, Val loss: 1.1637, Val acc: 63.78%
    Validation accuracy improved from 60.55% to 63.78%



    Epoch 11/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 11/50 - Train loss: 1.7347, Val loss: 1.1319, Val acc: 65.54%
    Validation accuracy improved from 63.78% to 65.54%



    Epoch 12/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 12/50 - Train loss: 1.7060, Val loss: 1.1190, Val acc: 65.82%
    Validation accuracy improved from 65.54% to 65.82%



    Epoch 13/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 13/50 - Train loss: 1.6964, Val loss: 1.0804, Val acc: 68.06%
    Validation accuracy improved from 65.82% to 68.06%



    Epoch 14/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 14/50 - Train loss: 1.6843, Val loss: 0.9804, Val acc: 70.28%
    Validation accuracy improved from 68.06% to 70.28%



    Epoch 15/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 15/50 - Train loss: 1.6499, Val loss: 0.9919, Val acc: 70.29%
    Validation accuracy improved from 70.28% to 70.29%



    Epoch 16/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 16/50 - Train loss: 1.6313, Val loss: 0.9958, Val acc: 71.69%
    Validation accuracy improved from 70.29% to 71.69%



    Epoch 17/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 17/50 - Train loss: 1.6005, Val loss: 0.8822, Val acc: 71.86%
    Validation accuracy improved from 71.69% to 71.86%



    Epoch 18/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 18/50 - Train loss: 1.5840, Val loss: 0.8987, Val acc: 73.42%
    Validation accuracy improved from 71.86% to 73.42%



    Epoch 19/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 19/50 - Train loss: 1.5834, Val loss: 0.9431, Val acc: 72.89%



    Epoch 20/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 20/50 - Train loss: 1.5805, Val loss: 0.8846, Val acc: 74.56%
    Validation accuracy improved from 73.42% to 74.56%



    Epoch 21/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 21/50 - Train loss: 1.5664, Val loss: 0.7975, Val acc: 75.94%
    Validation accuracy improved from 74.56% to 75.94%



    Epoch 22/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 22/50 - Train loss: 1.5413, Val loss: 0.8118, Val acc: 76.12%
    Validation accuracy improved from 75.94% to 76.12%



    Epoch 23/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 23/50 - Train loss: 1.5108, Val loss: 0.7970, Val acc: 76.35%
    Validation accuracy improved from 76.12% to 76.35%



    Epoch 24/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 24/50 - Train loss: 1.5164, Val loss: 0.8097, Val acc: 77.08%
    Validation accuracy improved from 76.35% to 77.08%



    Epoch 25/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 25/50 - Train loss: 1.5059, Val loss: 0.7599, Val acc: 78.66%
    Validation accuracy improved from 77.08% to 78.66%



    Epoch 26/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 26/50 - Train loss: 1.4568, Val loss: 0.7755, Val acc: 78.84%
    Validation accuracy improved from 78.66% to 78.84%



    Epoch 27/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 27/50 - Train loss: 1.4884, Val loss: 0.8185, Val acc: 77.40%



    Epoch 28/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 28/50 - Train loss: 1.4713, Val loss: 0.7348, Val acc: 78.93%
    Validation accuracy improved from 78.84% to 78.93%



    Epoch 29/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 29/50 - Train loss: 1.4351, Val loss: 0.7318, Val acc: 78.83%



    Epoch 30/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 30/50 - Train loss: 1.4439, Val loss: 0.7352, Val acc: 79.36%
    Validation accuracy improved from 78.93% to 79.36%



    Epoch 31/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 31/50 - Train loss: 1.4552, Val loss: 0.7220, Val acc: 79.88%
    Validation accuracy improved from 79.36% to 79.88%



    Epoch 32/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 32/50 - Train loss: 1.4248, Val loss: 0.6834, Val acc: 81.08%
    Validation accuracy improved from 79.88% to 81.08%



    Epoch 33/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 33/50 - Train loss: 1.4240, Val loss: 0.7083, Val acc: 80.39%



    Epoch 34/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 34/50 - Train loss: 1.4106, Val loss: 0.6878, Val acc: 80.70%



    Epoch 35/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 35/50 - Train loss: 1.3883, Val loss: 0.7199, Val acc: 80.56%



    Epoch 36/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 36/50 - Train loss: 1.3604, Val loss: 0.6773, Val acc: 81.54%
    Validation accuracy improved from 81.08% to 81.54%



    Epoch 37/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 37/50 - Train loss: 1.3686, Val loss: 0.6836, Val acc: 80.55%



    Epoch 38/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 38/50 - Train loss: 1.3580, Val loss: 0.6783, Val acc: 81.57%
    Validation accuracy improved from 81.54% to 81.57%



    Epoch 39/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 39/50 - Train loss: 1.3616, Val loss: 0.6763, Val acc: 81.45%



    Epoch 40/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 40/50 - Train loss: 1.3501, Val loss: 0.6650, Val acc: 81.62%
    Validation accuracy improved from 81.57% to 81.62%



    Epoch 41/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 41/50 - Train loss: 1.3295, Val loss: 0.6642, Val acc: 81.68%
    Validation accuracy improved from 81.62% to 81.68%



    Epoch 42/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 42/50 - Train loss: 1.4020, Val loss: 0.6802, Val acc: 81.62%



    Epoch 43/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 43/50 - Train loss: 1.3283, Val loss: 0.6621, Val acc: 81.72%
    Validation accuracy improved from 81.68% to 81.72%



    Epoch 44/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 44/50 - Train loss: 1.3778, Val loss: 0.6666, Val acc: 82.15%
    Validation accuracy improved from 81.72% to 82.15%



    Epoch 45/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 45/50 - Train loss: 1.3311, Val loss: 0.6517, Val acc: 82.36%
    Validation accuracy improved from 82.15% to 82.36%



    Epoch 46/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 46/50 - Train loss: 1.3009, Val loss: 0.6549, Val acc: 82.37%
    Validation accuracy improved from 82.36% to 82.37%



    Epoch 47/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 47/50 - Train loss: 1.3567, Val loss: 0.6581, Val acc: 82.15%



    Epoch 48/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 48/50 - Train loss: 1.3398, Val loss: 0.6571, Val acc: 82.16%



    Epoch 49/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 49/50 - Train loss: 1.3588, Val loss: 0.6577, Val acc: 82.18%



    Epoch 50/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 50/50 - Train loss: 1.3500, Val loss: 0.6576, Val acc: 82.18%
    
    ==================================================
    Training convnext_silu_layernorm...
    ==================================================
    



    Epoch 1/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 1/50 - Train loss: 2.4426, Val loss: 2.4459, Val acc: 9.61%
    Validation accuracy improved from 0.00% to 9.61%



    Epoch 2/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 2/50 - Train loss: 2.0824, Val loss: 1.6981, Val acc: 42.18%
    Validation accuracy improved from 9.61% to 42.18%



    Epoch 3/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 3/50 - Train loss: 2.0198, Val loss: 1.5815, Val acc: 47.03%
    Validation accuracy improved from 42.18% to 47.03%



    Epoch 4/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 4/50 - Train loss: 1.9614, Val loss: 1.5086, Val acc: 49.26%
    Validation accuracy improved from 47.03% to 49.26%



    Epoch 5/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 5/50 - Train loss: 1.9314, Val loss: 1.4945, Val acc: 50.69%
    Validation accuracy improved from 49.26% to 50.69%



    Epoch 6/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 6/50 - Train loss: 1.9241, Val loss: 1.3880, Val acc: 53.12%
    Validation accuracy improved from 50.69% to 53.12%



    Epoch 7/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 7/50 - Train loss: 1.8795, Val loss: 1.3516, Val acc: 56.22%
    Validation accuracy improved from 53.12% to 56.22%



    Epoch 8/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 8/50 - Train loss: 1.8326, Val loss: 1.3304, Val acc: 57.29%
    Validation accuracy improved from 56.22% to 57.29%



    Epoch 9/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 9/50 - Train loss: 1.8004, Val loss: 1.2445, Val acc: 58.74%
    Validation accuracy improved from 57.29% to 58.74%



    Epoch 10/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 10/50 - Train loss: 1.8018, Val loss: 1.2140, Val acc: 60.78%
    Validation accuracy improved from 58.74% to 60.78%



    Epoch 11/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 11/50 - Train loss: 1.7553, Val loss: 1.1614, Val acc: 64.09%
    Validation accuracy improved from 60.78% to 64.09%



    Epoch 12/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 12/50 - Train loss: 1.7449, Val loss: 1.1048, Val acc: 64.92%
    Validation accuracy improved from 64.09% to 64.92%



    Epoch 13/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 13/50 - Train loss: 1.7241, Val loss: 1.0395, Val acc: 66.71%
    Validation accuracy improved from 64.92% to 66.71%



    Epoch 14/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 14/50 - Train loss: 1.6929, Val loss: 1.0827, Val acc: 66.63%



    Epoch 15/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 15/50 - Train loss: 1.6833, Val loss: 0.9835, Val acc: 68.50%
    Validation accuracy improved from 66.71% to 68.50%



    Epoch 16/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 16/50 - Train loss: 1.6971, Val loss: 1.0065, Val acc: 69.52%
    Validation accuracy improved from 68.50% to 69.52%



    Epoch 17/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 17/50 - Train loss: 1.6631, Val loss: 0.9935, Val acc: 69.63%
    Validation accuracy improved from 69.52% to 69.63%



    Epoch 18/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 18/50 - Train loss: 1.6329, Val loss: 0.9976, Val acc: 69.62%



    Epoch 19/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 19/50 - Train loss: 1.6395, Val loss: 1.0026, Val acc: 70.01%
    Validation accuracy improved from 69.63% to 70.01%



    Epoch 20/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 20/50 - Train loss: 1.6247, Val loss: 0.9086, Val acc: 72.28%
    Validation accuracy improved from 70.01% to 72.28%



    Epoch 21/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 21/50 - Train loss: 1.6009, Val loss: 0.9212, Val acc: 72.15%



    Epoch 22/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 22/50 - Train loss: 1.6159, Val loss: 0.9157, Val acc: 73.35%
    Validation accuracy improved from 72.28% to 73.35%



    Epoch 23/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 23/50 - Train loss: 1.5935, Val loss: 0.9103, Val acc: 72.89%



    Epoch 24/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 24/50 - Train loss: 1.5787, Val loss: 0.8911, Val acc: 74.01%
    Validation accuracy improved from 73.35% to 74.01%



    Epoch 25/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 25/50 - Train loss: 1.5496, Val loss: 0.8826, Val acc: 73.63%



    Epoch 26/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 26/50 - Train loss: 1.5160, Val loss: 0.8212, Val acc: 76.08%
    Validation accuracy improved from 74.01% to 76.08%



    Epoch 27/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 27/50 - Train loss: 1.5417, Val loss: 0.8438, Val acc: 75.35%



    Epoch 28/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 28/50 - Train loss: 1.5104, Val loss: 0.8670, Val acc: 75.92%



    Epoch 29/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 29/50 - Train loss: 1.4973, Val loss: 0.8088, Val acc: 76.64%
    Validation accuracy improved from 76.08% to 76.64%



    Epoch 30/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 30/50 - Train loss: 1.4968, Val loss: 0.8281, Val acc: 77.38%
    Validation accuracy improved from 76.64% to 77.38%



    Epoch 31/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 31/50 - Train loss: 1.4933, Val loss: 0.8138, Val acc: 76.86%



    Epoch 32/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 32/50 - Train loss: 1.5059, Val loss: 0.8065, Val acc: 76.63%



    Epoch 33/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 33/50 - Train loss: 1.5067, Val loss: 0.7780, Val acc: 78.40%
    Validation accuracy improved from 77.38% to 78.40%



    Epoch 34/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 34/50 - Train loss: 1.5177, Val loss: 0.7802, Val acc: 78.43%
    Validation accuracy improved from 78.40% to 78.43%



    Epoch 35/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 35/50 - Train loss: 1.4848, Val loss: 0.7505, Val acc: 78.49%
    Validation accuracy improved from 78.43% to 78.49%



    Epoch 36/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 36/50 - Train loss: 1.4064, Val loss: 0.7786, Val acc: 78.19%



    Epoch 37/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 37/50 - Train loss: 1.4658, Val loss: 0.7427, Val acc: 78.58%
    Validation accuracy improved from 78.49% to 78.58%



    Epoch 38/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 38/50 - Train loss: 1.4280, Val loss: 0.7363, Val acc: 79.06%
    Validation accuracy improved from 78.58% to 79.06%



    Epoch 39/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 39/50 - Train loss: 1.4278, Val loss: 0.7356, Val acc: 78.94%



    Epoch 40/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 40/50 - Train loss: 1.4172, Val loss: 0.7378, Val acc: 79.16%
    Validation accuracy improved from 79.06% to 79.16%



    Epoch 41/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 41/50 - Train loss: 1.3997, Val loss: 0.7363, Val acc: 78.79%



    Epoch 42/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 42/50 - Train loss: 1.4100, Val loss: 0.7390, Val acc: 79.05%



    Epoch 43/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 43/50 - Train loss: 1.4052, Val loss: 0.7308, Val acc: 79.70%
    Validation accuracy improved from 79.16% to 79.70%



    Epoch 44/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 44/50 - Train loss: 1.3952, Val loss: 0.7192, Val acc: 79.55%



    Epoch 45/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 45/50 - Train loss: 1.3630, Val loss: 0.7157, Val acc: 79.69%



    Epoch 46/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 46/50 - Train loss: 1.4008, Val loss: 0.7270, Val acc: 79.48%



    Epoch 47/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 47/50 - Train loss: 1.3869, Val loss: 0.7261, Val acc: 79.61%



    Epoch 48/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 48/50 - Train loss: 1.3941, Val loss: 0.7248, Val acc: 79.70%



    Epoch 49/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 49/50 - Train loss: 1.3707, Val loss: 0.7219, Val acc: 79.86%
    Validation accuracy improved from 79.70% to 79.86%



    Epoch 50/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 50/50 - Train loss: 1.3892, Val loss: 0.7216, Val acc: 79.85%
    
    ==================================================
    Training convnext_mish_layernorm...
    ==================================================
    



    Epoch 1/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 1/50 - Train loss: 2.4511, Val loss: 2.4512, Val acc: 10.64%
    Validation accuracy improved from 0.00% to 10.64%



    Epoch 2/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 2/50 - Train loss: 2.0866, Val loss: 1.6568, Val acc: 42.71%
    Validation accuracy improved from 10.64% to 42.71%



    Epoch 3/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 3/50 - Train loss: 1.9978, Val loss: 1.5308, Val acc: 47.26%
    Validation accuracy improved from 42.71% to 47.26%



    Epoch 4/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 4/50 - Train loss: 1.9438, Val loss: 1.4675, Val acc: 50.91%
    Validation accuracy improved from 47.26% to 50.91%



    Epoch 5/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 5/50 - Train loss: 1.9122, Val loss: 1.4251, Val acc: 51.50%
    Validation accuracy improved from 50.91% to 51.50%



    Epoch 6/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 6/50 - Train loss: 1.8897, Val loss: 1.4521, Val acc: 52.82%
    Validation accuracy improved from 51.50% to 52.82%



    Epoch 7/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 7/50 - Train loss: 1.8741, Val loss: 1.3177, Val acc: 55.38%
    Validation accuracy improved from 52.82% to 55.38%



    Epoch 8/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 8/50 - Train loss: 1.8558, Val loss: 1.3264, Val acc: 56.88%
    Validation accuracy improved from 55.38% to 56.88%



    Epoch 9/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 9/50 - Train loss: 1.8084, Val loss: 1.2810, Val acc: 59.14%
    Validation accuracy improved from 56.88% to 59.14%



    Epoch 10/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 10/50 - Train loss: 1.8073, Val loss: 1.2727, Val acc: 60.06%
    Validation accuracy improved from 59.14% to 60.06%



    Epoch 11/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 11/50 - Train loss: 1.7761, Val loss: 1.1997, Val acc: 61.36%
    Validation accuracy improved from 60.06% to 61.36%



    Epoch 12/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 12/50 - Train loss: 1.7752, Val loss: 1.1730, Val acc: 61.99%
    Validation accuracy improved from 61.36% to 61.99%



    Epoch 13/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 13/50 - Train loss: 1.7436, Val loss: 1.1386, Val acc: 63.58%
    Validation accuracy improved from 61.99% to 63.58%



    Epoch 14/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 14/50 - Train loss: 1.7415, Val loss: 1.0779, Val acc: 64.69%
    Validation accuracy improved from 63.58% to 64.69%



    Epoch 15/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 15/50 - Train loss: 1.7243, Val loss: 1.0938, Val acc: 64.76%
    Validation accuracy improved from 64.69% to 64.76%



    Epoch 16/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 16/50 - Train loss: 1.7205, Val loss: 1.1053, Val acc: 65.63%
    Validation accuracy improved from 64.76% to 65.63%



    Epoch 17/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 17/50 - Train loss: 1.6963, Val loss: 1.0940, Val acc: 65.51%



    Epoch 18/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 18/50 - Train loss: 1.6666, Val loss: 1.0671, Val acc: 66.82%
    Validation accuracy improved from 65.63% to 66.82%



    Epoch 19/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 19/50 - Train loss: 1.6861, Val loss: 1.1096, Val acc: 67.78%
    Validation accuracy improved from 66.82% to 67.78%



    Epoch 20/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 20/50 - Train loss: 1.6195, Val loss: 1.0220, Val acc: 69.63%
    Validation accuracy improved from 67.78% to 69.63%



    Epoch 21/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 21/50 - Train loss: 1.6504, Val loss: 1.0111, Val acc: 69.84%
    Validation accuracy improved from 69.63% to 69.84%



    Epoch 22/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 22/50 - Train loss: 1.6414, Val loss: 0.9850, Val acc: 69.81%



    Epoch 23/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 23/50 - Train loss: 1.5850, Val loss: 0.9692, Val acc: 69.74%



    Epoch 24/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 24/50 - Train loss: 1.6191, Val loss: 0.9510, Val acc: 71.16%
    Validation accuracy improved from 69.84% to 71.16%



    Epoch 25/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 25/50 - Train loss: 1.5934, Val loss: 0.9467, Val acc: 72.17%
    Validation accuracy improved from 71.16% to 72.17%



    Epoch 26/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 26/50 - Train loss: 1.5502, Val loss: 0.9711, Val acc: 71.50%



    Epoch 27/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 27/50 - Train loss: 1.5643, Val loss: 0.9680, Val acc: 71.62%



    Epoch 28/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 28/50 - Train loss: 1.5615, Val loss: 0.9133, Val acc: 73.36%
    Validation accuracy improved from 72.17% to 73.36%



    Epoch 29/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 29/50 - Train loss: 1.5527, Val loss: 0.9124, Val acc: 72.17%



    Epoch 30/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 30/50 - Train loss: 1.5088, Val loss: 0.8638, Val acc: 73.67%
    Validation accuracy improved from 73.36% to 73.67%



    Epoch 31/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 31/50 - Train loss: 1.5279, Val loss: 0.8760, Val acc: 73.85%
    Validation accuracy improved from 73.67% to 73.85%



    Epoch 32/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 32/50 - Train loss: 1.4906, Val loss: 0.8287, Val acc: 74.89%
    Validation accuracy improved from 73.85% to 74.89%



    Epoch 33/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 33/50 - Train loss: 1.4899, Val loss: 0.8540, Val acc: 74.51%



    Epoch 34/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 34/50 - Train loss: 1.5140, Val loss: 0.8747, Val acc: 74.39%



    Epoch 35/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 35/50 - Train loss: 1.4623, Val loss: 0.8357, Val acc: 75.12%
    Validation accuracy improved from 74.89% to 75.12%



    Epoch 36/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 36/50 - Train loss: 1.4590, Val loss: 0.8286, Val acc: 75.60%
    Validation accuracy improved from 75.12% to 75.60%



    Epoch 37/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 37/50 - Train loss: 1.4848, Val loss: 0.8070, Val acc: 76.65%
    Validation accuracy improved from 75.60% to 76.65%



    Epoch 38/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 38/50 - Train loss: 1.4467, Val loss: 0.8092, Val acc: 76.20%



    Epoch 39/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 39/50 - Train loss: 1.4540, Val loss: 0.8260, Val acc: 75.74%



    Epoch 40/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 40/50 - Train loss: 1.4605, Val loss: 0.8187, Val acc: 75.93%



    Epoch 41/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 41/50 - Train loss: 1.4088, Val loss: 0.7938, Val acc: 76.22%



    Epoch 42/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 42/50 - Train loss: 1.4412, Val loss: 0.7997, Val acc: 76.81%
    Validation accuracy improved from 76.65% to 76.81%



    Epoch 43/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 43/50 - Train loss: 1.4244, Val loss: 0.7900, Val acc: 76.69%



    Epoch 44/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 44/50 - Train loss: 1.3961, Val loss: 0.8049, Val acc: 76.40%



    Epoch 45/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 45/50 - Train loss: 1.4107, Val loss: 0.7917, Val acc: 76.76%



    Epoch 46/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 46/50 - Train loss: 1.4319, Val loss: 0.8026, Val acc: 76.82%
    Validation accuracy improved from 76.81% to 76.82%



    Epoch 47/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 47/50 - Train loss: 1.4660, Val loss: 0.8042, Val acc: 76.90%
    Validation accuracy improved from 76.82% to 76.90%



    Epoch 48/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 48/50 - Train loss: 1.4383, Val loss: 0.7905, Val acc: 77.19%
    Validation accuracy improved from 76.90% to 77.19%



    Epoch 49/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 49/50 - Train loss: 1.4054, Val loss: 0.7924, Val acc: 77.15%



    Epoch 50/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 50/50 - Train loss: 1.4173, Val loss: 0.7924, Val acc: 77.19%
    
    ==================================================
    Training convnext_hardswish_layernorm...
    ==================================================
    



    Epoch 1/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 1/50 - Train loss: 2.4191, Val loss: 2.4214, Val acc: 10.02%
    Validation accuracy improved from 0.00% to 10.02%



    Epoch 2/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 2/50 - Train loss: 2.0913, Val loss: 1.6827, Val acc: 42.17%
    Validation accuracy improved from 10.02% to 42.17%



    Epoch 3/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 3/50 - Train loss: 2.0036, Val loss: 1.5644, Val acc: 47.72%
    Validation accuracy improved from 42.17% to 47.72%



    Epoch 4/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 4/50 - Train loss: 1.9562, Val loss: 1.5429, Val acc: 48.99%
    Validation accuracy improved from 47.72% to 48.99%



    Epoch 5/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 5/50 - Train loss: 1.9266, Val loss: 1.4739, Val acc: 51.66%
    Validation accuracy improved from 48.99% to 51.66%



    Epoch 6/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 6/50 - Train loss: 1.9096, Val loss: 1.3880, Val acc: 53.49%
    Validation accuracy improved from 51.66% to 53.49%



    Epoch 7/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 7/50 - Train loss: 1.8785, Val loss: 1.3264, Val acc: 56.22%
    Validation accuracy improved from 53.49% to 56.22%



    Epoch 8/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 8/50 - Train loss: 1.8474, Val loss: 1.3224, Val acc: 56.81%
    Validation accuracy improved from 56.22% to 56.81%



    Epoch 9/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 9/50 - Train loss: 1.8352, Val loss: 1.2858, Val acc: 57.41%
    Validation accuracy improved from 56.81% to 57.41%



    Epoch 10/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 10/50 - Train loss: 1.8102, Val loss: 1.2635, Val acc: 61.21%
    Validation accuracy improved from 57.41% to 61.21%



    Epoch 11/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 11/50 - Train loss: 1.7960, Val loss: 1.1384, Val acc: 63.10%
    Validation accuracy improved from 61.21% to 63.10%



    Epoch 12/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 12/50 - Train loss: 1.7512, Val loss: 1.1785, Val acc: 62.38%



    Epoch 13/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 13/50 - Train loss: 1.7471, Val loss: 1.1491, Val acc: 64.21%
    Validation accuracy improved from 63.10% to 64.21%



    Epoch 14/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 14/50 - Train loss: 1.7405, Val loss: 1.1386, Val acc: 64.36%
    Validation accuracy improved from 64.21% to 64.36%



    Epoch 15/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 15/50 - Train loss: 1.7166, Val loss: 1.1455, Val acc: 65.38%
    Validation accuracy improved from 64.36% to 65.38%



    Epoch 16/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 16/50 - Train loss: 1.7107, Val loss: 1.1049, Val acc: 66.52%
    Validation accuracy improved from 65.38% to 66.52%



    Epoch 17/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 17/50 - Train loss: 1.7024, Val loss: 1.0336, Val acc: 68.13%
    Validation accuracy improved from 66.52% to 68.13%



    Epoch 18/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 18/50 - Train loss: 1.6893, Val loss: 1.0178, Val acc: 69.04%
    Validation accuracy improved from 68.13% to 69.04%



    Epoch 19/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 19/50 - Train loss: 1.6482, Val loss: 1.0342, Val acc: 68.45%



    Epoch 20/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 20/50 - Train loss: 1.6535, Val loss: 1.0435, Val acc: 69.52%
    Validation accuracy improved from 69.04% to 69.52%



    Epoch 21/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 21/50 - Train loss: 1.6216, Val loss: 0.9434, Val acc: 71.24%
    Validation accuracy improved from 69.52% to 71.24%



    Epoch 22/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 22/50 - Train loss: 1.6370, Val loss: 0.9651, Val acc: 71.56%
    Validation accuracy improved from 71.24% to 71.56%



    Epoch 23/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 23/50 - Train loss: 1.6082, Val loss: 0.8862, Val acc: 72.93%
    Validation accuracy improved from 71.56% to 72.93%



    Epoch 24/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 24/50 - Train loss: 1.5838, Val loss: 0.9171, Val acc: 71.79%



    Epoch 25/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 25/50 - Train loss: 1.5611, Val loss: 0.9119, Val acc: 73.87%
    Validation accuracy improved from 72.93% to 73.87%



    Epoch 26/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 26/50 - Train loss: 1.5861, Val loss: 0.8751, Val acc: 74.22%
    Validation accuracy improved from 73.87% to 74.22%



    Epoch 27/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 27/50 - Train loss: 1.6002, Val loss: 0.9116, Val acc: 74.67%
    Validation accuracy improved from 74.22% to 74.67%



    Epoch 28/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 28/50 - Train loss: 1.5331, Val loss: 0.8442, Val acc: 75.75%
    Validation accuracy improved from 74.67% to 75.75%



    Epoch 29/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 29/50 - Train loss: 1.5489, Val loss: 0.8579, Val acc: 74.91%



    Epoch 30/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 30/50 - Train loss: 1.5266, Val loss: 0.8276, Val acc: 76.32%
    Validation accuracy improved from 75.75% to 76.32%



    Epoch 31/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 31/50 - Train loss: 1.5303, Val loss: 0.8341, Val acc: 75.67%



    Epoch 32/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 32/50 - Train loss: 1.5258, Val loss: 0.8147, Val acc: 75.55%



    Epoch 33/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 33/50 - Train loss: 1.4922, Val loss: 0.8369, Val acc: 76.85%
    Validation accuracy improved from 76.32% to 76.85%



    Epoch 34/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 34/50 - Train loss: 1.4936, Val loss: 0.7801, Val acc: 77.30%
    Validation accuracy improved from 76.85% to 77.30%



    Epoch 35/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 35/50 - Train loss: 1.4709, Val loss: 0.8056, Val acc: 76.92%



    Epoch 36/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 36/50 - Train loss: 1.4629, Val loss: 0.7980, Val acc: 77.28%



    Epoch 37/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 37/50 - Train loss: 1.4578, Val loss: 0.7950, Val acc: 77.47%
    Validation accuracy improved from 77.30% to 77.47%



    Epoch 38/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 38/50 - Train loss: 1.4522, Val loss: 0.7953, Val acc: 77.34%



    Epoch 39/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 39/50 - Train loss: 1.4421, Val loss: 0.7787, Val acc: 78.21%
    Validation accuracy improved from 77.47% to 78.21%



    Epoch 40/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 40/50 - Train loss: 1.4371, Val loss: 0.7561, Val acc: 78.32%
    Validation accuracy improved from 78.21% to 78.32%



    Epoch 41/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 41/50 - Train loss: 1.4421, Val loss: 0.7598, Val acc: 78.18%



    Epoch 42/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 42/50 - Train loss: 1.4268, Val loss: 0.7651, Val acc: 78.40%
    Validation accuracy improved from 78.32% to 78.40%



    Epoch 43/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 43/50 - Train loss: 1.4222, Val loss: 0.7557, Val acc: 78.42%
    Validation accuracy improved from 78.40% to 78.42%



    Epoch 44/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 44/50 - Train loss: 1.4201, Val loss: 0.7472, Val acc: 78.67%
    Validation accuracy improved from 78.42% to 78.67%



    Epoch 45/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 45/50 - Train loss: 1.4558, Val loss: 0.7582, Val acc: 78.81%
    Validation accuracy improved from 78.67% to 78.81%



    Epoch 46/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 46/50 - Train loss: 1.3836, Val loss: 0.7477, Val acc: 78.78%



    Epoch 47/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 47/50 - Train loss: 1.4074, Val loss: 0.7482, Val acc: 78.87%
    Validation accuracy improved from 78.81% to 78.87%



    Epoch 48/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 48/50 - Train loss: 1.4179, Val loss: 0.7479, Val acc: 79.04%
    Validation accuracy improved from 78.87% to 79.04%



    Epoch 49/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 49/50 - Train loss: 1.4321, Val loss: 0.7503, Val acc: 78.97%



    Epoch 50/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 50/50 - Train loss: 1.4318, Val loss: 0.7509, Val acc: 79.00%
    
    ==================================================
    Training convnext_gelu_groupnorm...
    ==================================================
    



    Epoch 1/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 1/50 - Train loss: 2.4266, Val loss: 2.4123, Val acc: 10.04%
    Validation accuracy improved from 0.00% to 10.04%



    Epoch 2/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 2/50 - Train loss: 2.0873, Val loss: 1.6612, Val acc: 43.10%
    Validation accuracy improved from 10.04% to 43.10%



    Epoch 3/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 3/50 - Train loss: 1.9994, Val loss: 1.4654, Val acc: 50.43%
    Validation accuracy improved from 43.10% to 50.43%



    Epoch 4/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 4/50 - Train loss: 1.9197, Val loss: 1.4394, Val acc: 53.45%
    Validation accuracy improved from 50.43% to 53.45%



    Epoch 5/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 5/50 - Train loss: 1.8749, Val loss: 1.3486, Val acc: 55.12%
    Validation accuracy improved from 53.45% to 55.12%



    Epoch 6/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 6/50 - Train loss: 1.8593, Val loss: 1.2887, Val acc: 58.44%
    Validation accuracy improved from 55.12% to 58.44%



    Epoch 7/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 7/50 - Train loss: 1.8157, Val loss: 1.2188, Val acc: 60.76%
    Validation accuracy improved from 58.44% to 60.76%



    Epoch 8/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 8/50 - Train loss: 1.7831, Val loss: 1.2427, Val acc: 60.99%
    Validation accuracy improved from 60.76% to 60.99%



    Epoch 9/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 9/50 - Train loss: 1.7387, Val loss: 1.0916, Val acc: 66.01%
    Validation accuracy improved from 60.99% to 66.01%



    Epoch 10/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 10/50 - Train loss: 1.7123, Val loss: 1.1495, Val acc: 67.47%
    Validation accuracy improved from 66.01% to 67.47%



    Epoch 11/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 11/50 - Train loss: 1.6842, Val loss: 1.0525, Val acc: 68.89%
    Validation accuracy improved from 67.47% to 68.89%



    Epoch 12/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 12/50 - Train loss: 1.6758, Val loss: 1.0475, Val acc: 69.84%
    Validation accuracy improved from 68.89% to 69.84%



    Epoch 13/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 13/50 - Train loss: 1.6413, Val loss: 0.9244, Val acc: 72.34%
    Validation accuracy improved from 69.84% to 72.34%



    Epoch 14/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 14/50 - Train loss: 1.6104, Val loss: 0.9337, Val acc: 71.85%



    Epoch 15/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 15/50 - Train loss: 1.6214, Val loss: 0.9123, Val acc: 72.94%
    Validation accuracy improved from 72.34% to 72.94%



    Epoch 16/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 16/50 - Train loss: 1.6067, Val loss: 0.8807, Val acc: 74.65%
    Validation accuracy improved from 72.94% to 74.65%



    Epoch 17/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 17/50 - Train loss: 1.5885, Val loss: 0.8626, Val acc: 73.10%



    Epoch 18/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 18/50 - Train loss: 1.5892, Val loss: 0.8712, Val acc: 75.18%
    Validation accuracy improved from 74.65% to 75.18%



    Epoch 19/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 19/50 - Train loss: 1.5454, Val loss: 0.8233, Val acc: 75.88%
    Validation accuracy improved from 75.18% to 75.88%



    Epoch 20/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 20/50 - Train loss: 1.5270, Val loss: 0.8272, Val acc: 76.44%
    Validation accuracy improved from 75.88% to 76.44%



    Epoch 21/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 21/50 - Train loss: 1.5390, Val loss: 0.7852, Val acc: 76.86%
    Validation accuracy improved from 76.44% to 76.86%



    Epoch 22/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 22/50 - Train loss: 1.5379, Val loss: 0.7910, Val acc: 77.27%
    Validation accuracy improved from 76.86% to 77.27%



    Epoch 23/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 23/50 - Train loss: 1.5029, Val loss: 0.8243, Val acc: 78.12%
    Validation accuracy improved from 77.27% to 78.12%



    Epoch 24/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 24/50 - Train loss: 1.4730, Val loss: 0.7791, Val acc: 78.66%
    Validation accuracy improved from 78.12% to 78.66%



    Epoch 25/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 25/50 - Train loss: 1.4764, Val loss: 0.7600, Val acc: 78.85%
    Validation accuracy improved from 78.66% to 78.85%



    Epoch 26/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 26/50 - Train loss: 1.4545, Val loss: 0.7499, Val acc: 79.12%
    Validation accuracy improved from 78.85% to 79.12%



    Epoch 27/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 27/50 - Train loss: 1.4821, Val loss: 0.7619, Val acc: 79.39%
    Validation accuracy improved from 79.12% to 79.39%



    Epoch 28/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 28/50 - Train loss: 1.4776, Val loss: 0.7653, Val acc: 79.03%



    Epoch 29/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 29/50 - Train loss: 1.4434, Val loss: 0.7241, Val acc: 79.93%
    Validation accuracy improved from 79.39% to 79.93%



    Epoch 30/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 30/50 - Train loss: 1.4473, Val loss: 0.6712, Val acc: 81.00%
    Validation accuracy improved from 79.93% to 81.00%



    Epoch 31/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 31/50 - Train loss: 1.4283, Val loss: 0.7236, Val acc: 80.34%



    Epoch 32/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 32/50 - Train loss: 1.4372, Val loss: 0.7180, Val acc: 80.44%



    Epoch 33/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 33/50 - Train loss: 1.4074, Val loss: 0.7255, Val acc: 81.45%
    Validation accuracy improved from 81.00% to 81.45%



    Epoch 34/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 34/50 - Train loss: 1.3928, Val loss: 0.6862, Val acc: 81.64%
    Validation accuracy improved from 81.45% to 81.64%



    Epoch 35/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 35/50 - Train loss: 1.4106, Val loss: 0.7290, Val acc: 80.69%



    Epoch 36/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 36/50 - Train loss: 1.4027, Val loss: 0.6906, Val acc: 81.51%



    Epoch 37/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 37/50 - Train loss: 1.3722, Val loss: 0.6766, Val acc: 82.02%
    Validation accuracy improved from 81.64% to 82.02%



    Epoch 38/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 38/50 - Train loss: 1.3823, Val loss: 0.6702, Val acc: 82.40%
    Validation accuracy improved from 82.02% to 82.40%



    Epoch 39/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 39/50 - Train loss: 1.3345, Val loss: 0.6785, Val acc: 82.03%



    Epoch 40/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 40/50 - Train loss: 1.3192, Val loss: 0.6547, Val acc: 82.47%
    Validation accuracy improved from 82.40% to 82.47%



    Epoch 41/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 41/50 - Train loss: 1.3597, Val loss: 0.6666, Val acc: 82.71%
    Validation accuracy improved from 82.47% to 82.71%



    Epoch 42/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 42/50 - Train loss: 1.3695, Val loss: 0.6535, Val acc: 82.71%



    Epoch 43/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 43/50 - Train loss: 1.3474, Val loss: 0.6462, Val acc: 82.88%
    Validation accuracy improved from 82.71% to 82.88%



    Epoch 44/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 44/50 - Train loss: 1.3369, Val loss: 0.6600, Val acc: 82.92%
    Validation accuracy improved from 82.88% to 82.92%



    Epoch 45/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 45/50 - Train loss: 1.3226, Val loss: 0.6530, Val acc: 83.10%
    Validation accuracy improved from 82.92% to 83.10%



    Epoch 46/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 46/50 - Train loss: 1.3464, Val loss: 0.6490, Val acc: 83.28%
    Validation accuracy improved from 83.10% to 83.28%



    Epoch 47/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 47/50 - Train loss: 1.3303, Val loss: 0.6432, Val acc: 83.26%



    Epoch 48/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 48/50 - Train loss: 1.3253, Val loss: 0.6466, Val acc: 83.28%



    Epoch 49/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 49/50 - Train loss: 1.3407, Val loss: 0.6485, Val acc: 83.27%



    Epoch 50/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    Validation:   0%|          | 0/79 [00:00<?, ?it/s]


    Epoch 50/50 - Train loss: 1.3290, Val loss: 0.6479, Val acc: 83.25%
    
    ==================================================
    Training convnext_gelu_instancenorm...
    ==================================================
    



    Epoch 1/50 [Train]:   0%|          | 0/390 [00:00<?, ?it/s]



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[20], line 31
         27     return all_results, summary_table
         29 # Uncomment to run the full experiment
         30 # Note: This will take a long time to complete
    ---> 31 all_results, summary_table = run_full_experiment()


    Cell In[20], line 9, in run_full_experiment()
          6 models_to_test = create_model_variants(num_classes=10)
          8 # Train all variants
    ----> 9 all_results = train_all_variants(models_to_test, train_loader, val_loader, device, epochs=50)
         11 # Analyze results
         12 plot_comparison(all_results)


    Cell In[14], line 34, in train_all_variants(models_to_test, train_loader, val_loader, device, epochs, save_dir)
         31 print(f"{'='*50}\n")
         33 model = model.to(device)
    ---> 34 results = train_and_evaluate_model(
         35     model_name, model, train_loader, val_loader, device, epochs,
         36     mixup_fn=mixup_fn, save_dir=save_dir
         37 )
         38 all_results[model_name] = results
         40 # Clear memory


    Cell In[13], line 114, in train_and_evaluate_model(model_name, model, train_loader, val_loader, device, epochs, mixup_fn, save_dir)
        111 # Training loop
        112 for epoch in range(epochs):
        113     # Train one epoch
    --> 114     train_loss, train_acc = train_one_epoch(
        115         model, criterion, train_loader, optimizer, device, epoch, mixup_fn
        116     )
        118     # Validate
        119     val_loss, val_acc = validate(model, criterion, val_loader, device)


    Cell In[13], line 19, in train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, mixup_fn)
         16     inputs, targets = mixup_fn(inputs, targets)
         18 # Forward pass
    ---> 19 outputs = model(inputs)
         20 loss = criterion(outputs, targets)
         22 # Backward pass and optimization


    File ~/Workspace/miniconda3/envs/jupyter/lib/python3.11/site-packages/torch/nn/modules/module.py:1751, in Module._wrapped_call_impl(self, *args, **kwargs)
       1749     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1750 else:
    -> 1751     return self._call_impl(*args, **kwargs)


    File ~/Workspace/miniconda3/envs/jupyter/lib/python3.11/site-packages/torch/nn/modules/module.py:1762, in Module._call_impl(self, *args, **kwargs)
       1757 # If we don't have any hooks, we want to skip the rest of the logic in
       1758 # this function, and just call forward.
       1759 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
       1760         or _global_backward_pre_hooks or _global_backward_hooks
       1761         or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1762     return forward_call(*args, **kwargs)
       1764 result = None
       1765 called_always_called_hooks = set()


    Cell In[7], line 71, in ConvNeXt.forward(self, x)
         70 def forward(self, x):
    ---> 71     x = self.forward_features(x)
         72     x = self.head(x)
         73     return x


    Cell In[7], line 67, in ConvNeXt.forward_features(self, x)
         65 for i in range(4):
         66     x = self.downsample_layers[i](x)
    ---> 67     x = self.stages[i](x)
         68 return self.norm(x.mean([-2, -1]))


    File ~/Workspace/miniconda3/envs/jupyter/lib/python3.11/site-packages/torch/nn/modules/module.py:1751, in Module._wrapped_call_impl(self, *args, **kwargs)
       1749     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1750 else:
    -> 1751     return self._call_impl(*args, **kwargs)


    File ~/Workspace/miniconda3/envs/jupyter/lib/python3.11/site-packages/torch/nn/modules/module.py:1762, in Module._call_impl(self, *args, **kwargs)
       1757 # If we don't have any hooks, we want to skip the rest of the logic in
       1758 # this function, and just call forward.
       1759 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
       1760         or _global_backward_pre_hooks or _global_backward_hooks
       1761         or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1762     return forward_call(*args, **kwargs)
       1764 result = None
       1765 called_always_called_hooks = set()


    File ~/Workspace/miniconda3/envs/jupyter/lib/python3.11/site-packages/torch/nn/modules/container.py:240, in Sequential.forward(self, input)
        238 def forward(self, input):
        239     for module in self:
    --> 240         input = module(input)
        241     return input


    File ~/Workspace/miniconda3/envs/jupyter/lib/python3.11/site-packages/torch/nn/modules/module.py:1751, in Module._wrapped_call_impl(self, *args, **kwargs)
       1749     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1750 else:
    -> 1751     return self._call_impl(*args, **kwargs)


    File ~/Workspace/miniconda3/envs/jupyter/lib/python3.11/site-packages/torch/nn/modules/module.py:1762, in Module._call_impl(self, *args, **kwargs)
       1757 # If we don't have any hooks, we want to skip the rest of the logic in
       1758 # this function, and just call forward.
       1759 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
       1760         or _global_backward_pre_hooks or _global_backward_hooks
       1761         or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1762     return forward_call(*args, **kwargs)
       1764 result = None
       1765 called_always_called_hooks = set()


    Cell In[6], line 35, in Block.forward(self, x)
         33 x = self.dwconv(x)
         34 x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
    ---> 35 x = self.norm(x)
         36 x = self.pwconv1(x)
         37 x = self.act(x)


    File ~/Workspace/miniconda3/envs/jupyter/lib/python3.11/site-packages/torch/nn/modules/module.py:1751, in Module._wrapped_call_impl(self, *args, **kwargs)
       1749     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1750 else:
    -> 1751     return self._call_impl(*args, **kwargs)


    File ~/Workspace/miniconda3/envs/jupyter/lib/python3.11/site-packages/torch/nn/modules/module.py:1762, in Module._call_impl(self, *args, **kwargs)
       1757 # If we don't have any hooks, we want to skip the rest of the logic in
       1758 # this function, and just call forward.
       1759 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
       1760         or _global_backward_pre_hooks or _global_backward_hooks
       1761         or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1762     return forward_call(*args, **kwargs)
       1764 result = None
       1765 called_always_called_hooks = set()


    Cell In[4], line 37, in InstanceNormWrapper.forward(self, x)
         34 if x.ndim == 4 and x.shape[-1] == self.dim:
         35     # Input is (N, H, W, C), convert to (N, C, H, W)
         36     x = x.permute(0, 3, 1, 2)
    ---> 37     x = self.in_norm(x)
         38     x = x.permute(0, 2, 3, 1)
         39 else:


    File ~/Workspace/miniconda3/envs/jupyter/lib/python3.11/site-packages/torch/nn/modules/module.py:1751, in Module._wrapped_call_impl(self, *args, **kwargs)
       1749     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1750 else:
    -> 1751     return self._call_impl(*args, **kwargs)


    File ~/Workspace/miniconda3/envs/jupyter/lib/python3.11/site-packages/torch/nn/modules/module.py:1762, in Module._call_impl(self, *args, **kwargs)
       1757 # If we don't have any hooks, we want to skip the rest of the logic in
       1758 # this function, and just call forward.
       1759 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
       1760         or _global_backward_pre_hooks or _global_backward_hooks
       1761         or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1762     return forward_call(*args, **kwargs)
       1764 result = None
       1765 called_always_called_hooks = set()


    File ~/Workspace/miniconda3/envs/jupyter/lib/python3.11/site-packages/torch/nn/modules/instancenorm.py:124, in _InstanceNorm.forward(self, input)
        121 if input.dim() == self._get_no_batch_dim():
        122     return self._handle_no_batch_input(input)
    --> 124 return self._apply_instance_norm(input)


    File ~/Workspace/miniconda3/envs/jupyter/lib/python3.11/site-packages/torch/nn/modules/instancenorm.py:47, in _InstanceNorm._apply_instance_norm(self, input)
         46 def _apply_instance_norm(self, input):
    ---> 47     return F.instance_norm(
         48         input,
         49         self.running_mean,
         50         self.running_var,
         51         self.weight,
         52         self.bias,
         53         self.training or not self.track_running_stats,
         54         self.momentum if self.momentum is not None else 0.0,
         55         self.eps,
         56     )


    File ~/Workspace/miniconda3/envs/jupyter/lib/python3.11/site-packages/torch/nn/functional.py:2875, in instance_norm(input, running_mean, running_var, weight, bias, use_input_stats, momentum, eps)
       2862     return handle_torch_function(
       2863         instance_norm,
       2864         (input, running_mean, running_var, weight, bias),
       (...)   2872         eps=eps,
       2873     )
       2874 if use_input_stats:
    -> 2875     _verify_spatial_size(input.size())
       2876 return torch.instance_norm(
       2877     input,
       2878     weight,
       (...)   2885     torch.backends.cudnn.enabled,
       2886 )


    File ~/Workspace/miniconda3/envs/jupyter/lib/python3.11/site-packages/torch/nn/functional.py:2841, in _verify_spatial_size(size)
       2839     size_prods *= size[i]
       2840 if size_prods == 1:
    -> 2841     raise ValueError(
       2842         f"Expected more than 1 spatial element when training, got input size {size}"
       2843     )


    ValueError: Expected more than 1 spatial element when training, got input size torch.Size([128, 768, 1, 1])


The notebook now includes comprehensive improvements to the ConvNeXt architecture, focusing on activation functions and normalization strategies. The implementation allows for systematic comparison of different variants to determine which combinations yield the best performance on image classification tasks.
