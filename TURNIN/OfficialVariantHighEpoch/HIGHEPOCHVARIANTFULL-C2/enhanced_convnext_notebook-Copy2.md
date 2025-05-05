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
    and gracefully handles small spatial dimensions
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
            
            # Check spatial dimensions - if too small for InstanceNorm, just pass through
            if x.size(2) <= 1 and x.size(3) <= 1:
                # For 1x1 spatial dimensions, skip InstanceNorm
                result = x
            else:
                result = self.in_norm(x)
                
            x = result.permute(0, 2, 3, 1)
        else:
            # Check spatial dimensions - if too small for InstanceNorm, just pass through
            if x.size(2) <= 1 and x.size(3) <= 1:
                # For 1x1 spatial dimensions, skip InstanceNorm
                x = x
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
        
        # At small spatial dimensions, just use LayerNorm
        if (x.ndim == 4 and x.shape[1] == self.dim and (x.size(2) <= 1 and x.size(3) <= 1)) or \
           (x.ndim == 4 and x.shape[-1] == self.dim and (x.size(1) <= 1 and x.size(2) <= 1)):
            return ln_out
            
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
batch_size = 64
num_workers = 6

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
epochs = 40
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
        
        # Different normalizations with GELU
        'convnext_gelu_groupnorm': convnext_tiny_group_norm(num_classes=num_classes, act_layer=nn.GELU),
        # 'convnext_gelu_instancenorm': convnext_tiny_instance_norm(num_classes=num_classes, act_layer=nn.GELU),
        # 'convnext_gelu_layerinstancenorm': convnext_tiny_layer_instance_norm(num_classes=num_classes, act_layer=nn.GELU),
        
        # # Best activation (SiLU) with different normalizations
        # 'convnext_silu_groupnorm': convnext_tiny_group_norm(num_classes=num_classes, act_layer=nn.SiLU),
        # 'convnext_silu_instancenorm': convnext_tiny_instance_norm(num_classes=num_classes, act_layer=nn.SiLU),
        # 'convnext_silu_layerinstancenorm': convnext_tiny_layer_instance_norm(num_classes=num_classes, act_layer=nn.SiLU),
    }
    return models_to_test

# [ADDED] Function to train all model variants
def train_all_variants(models_to_test, train_loader, val_loader, device, epochs=40, save_dir='./checkpoints'):
    """Train and evaluate all model variants"""
    all_results = {}
    
    for model_name, model in models_to_test.items():
        print(f"\n{'='*40}")
        print(f"Training {model_name}...")
        print(f"{'='*40}\n")
        
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


    Epoch 1/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 1/40 - Train loss: 2.4161, Val loss: 2.4062, Val acc: 9.93%
    Validation accuracy improved from 0.00% to 9.93%



    Epoch 2/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 2/40 - Train loss: 2.0623, Val loss: 1.6408, Val acc: 44.18%
    Validation accuracy improved from 9.93% to 44.18%



    Epoch 3/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 3/40 - Train loss: 1.9746, Val loss: 1.4990, Val acc: 49.23%
    Validation accuracy improved from 44.18% to 49.23%



    Epoch 4/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 4/40 - Train loss: 1.9287, Val loss: 1.4450, Val acc: 50.75%
    Validation accuracy improved from 49.23% to 50.75%



    Epoch 5/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 5/40 - Train loss: 1.9064, Val loss: 1.4709, Val acc: 51.78%
    Validation accuracy improved from 50.75% to 51.78%



    Epoch 6/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 6/40 - Train loss: 1.8755, Val loss: 1.3975, Val acc: 55.24%
    Validation accuracy improved from 51.78% to 55.24%



    Epoch 7/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 7/40 - Train loss: 1.8351, Val loss: 1.2857, Val acc: 58.62%
    Validation accuracy improved from 55.24% to 58.62%



    Epoch 8/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 8/40 - Train loss: 1.7997, Val loss: 1.2221, Val acc: 61.90%
    Validation accuracy improved from 58.62% to 61.90%



    Epoch 9/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 9/40 - Train loss: 1.7840, Val loss: 1.1869, Val acc: 63.06%
    Validation accuracy improved from 61.90% to 63.06%



    Epoch 10/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 10/40 - Train loss: 1.7644, Val loss: 1.1827, Val acc: 63.84%
    Validation accuracy improved from 63.06% to 63.84%



    Epoch 11/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 11/40 - Train loss: 1.7367, Val loss: 1.1177, Val acc: 63.32%



    Epoch 12/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 12/40 - Train loss: 1.7105, Val loss: 1.1170, Val acc: 65.04%
    Validation accuracy improved from 63.84% to 65.04%



    Epoch 13/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 13/40 - Train loss: 1.6930, Val loss: 1.0527, Val acc: 67.27%
    Validation accuracy improved from 65.04% to 67.27%



    Epoch 14/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 14/40 - Train loss: 1.6854, Val loss: 1.0371, Val acc: 67.63%
    Validation accuracy improved from 67.27% to 67.63%



    Epoch 15/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 15/40 - Train loss: 1.6764, Val loss: 1.0725, Val acc: 68.91%
    Validation accuracy improved from 67.63% to 68.91%



    Epoch 16/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 16/40 - Train loss: 1.6511, Val loss: 1.0000, Val acc: 69.94%
    Validation accuracy improved from 68.91% to 69.94%



    Epoch 17/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 17/40 - Train loss: 1.6349, Val loss: 0.9556, Val acc: 70.57%
    Validation accuracy improved from 69.94% to 70.57%



    Epoch 18/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 18/40 - Train loss: 1.6178, Val loss: 0.9863, Val acc: 71.56%
    Validation accuracy improved from 70.57% to 71.56%



    Epoch 19/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 19/40 - Train loss: 1.6037, Val loss: 0.8893, Val acc: 73.48%
    Validation accuracy improved from 71.56% to 73.48%



    Epoch 20/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 20/40 - Train loss: 1.5810, Val loss: 0.9324, Val acc: 73.04%



    Epoch 21/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 21/40 - Train loss: 1.5826, Val loss: 0.8818, Val acc: 74.44%
    Validation accuracy improved from 73.48% to 74.44%



    Epoch 22/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 22/40 - Train loss: 1.5561, Val loss: 0.8449, Val acc: 74.89%
    Validation accuracy improved from 74.44% to 74.89%



    Epoch 23/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 23/40 - Train loss: 1.5749, Val loss: 0.8604, Val acc: 75.55%
    Validation accuracy improved from 74.89% to 75.55%



    Epoch 24/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 24/40 - Train loss: 1.5231, Val loss: 0.8480, Val acc: 75.91%
    Validation accuracy improved from 75.55% to 75.91%



    Epoch 25/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 25/40 - Train loss: 1.5287, Val loss: 0.8523, Val acc: 76.54%
    Validation accuracy improved from 75.91% to 76.54%



    Epoch 26/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 26/40 - Train loss: 1.4972, Val loss: 0.8041, Val acc: 77.11%
    Validation accuracy improved from 76.54% to 77.11%



    Epoch 27/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 27/40 - Train loss: 1.4933, Val loss: 0.7870, Val acc: 77.39%
    Validation accuracy improved from 77.11% to 77.39%



    Epoch 28/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 28/40 - Train loss: 1.4766, Val loss: 0.7888, Val acc: 78.18%
    Validation accuracy improved from 77.39% to 78.18%



    Epoch 29/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 29/40 - Train loss: 1.4698, Val loss: 0.7925, Val acc: 78.14%



    Epoch 30/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 30/40 - Train loss: 1.4780, Val loss: 0.7507, Val acc: 78.71%
    Validation accuracy improved from 78.18% to 78.71%



    Epoch 31/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 31/40 - Train loss: 1.4555, Val loss: 0.7672, Val acc: 79.06%
    Validation accuracy improved from 78.71% to 79.06%



    Epoch 32/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 32/40 - Train loss: 1.4472, Val loss: 0.7561, Val acc: 79.34%
    Validation accuracy improved from 79.06% to 79.34%



    Epoch 33/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 33/40 - Train loss: 1.4383, Val loss: 0.7294, Val acc: 79.44%
    Validation accuracy improved from 79.34% to 79.44%



    Epoch 34/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 34/40 - Train loss: 1.4289, Val loss: 0.7465, Val acc: 79.23%



    Epoch 35/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 35/40 - Train loss: 1.4241, Val loss: 0.7391, Val acc: 79.55%
    Validation accuracy improved from 79.44% to 79.55%



    Epoch 36/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 36/40 - Train loss: 1.4236, Val loss: 0.7193, Val acc: 79.82%
    Validation accuracy improved from 79.55% to 79.82%



    Epoch 37/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 37/40 - Train loss: 1.4000, Val loss: 0.7299, Val acc: 79.75%



    Epoch 38/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 38/40 - Train loss: 1.3998, Val loss: 0.7170, Val acc: 80.15%
    Validation accuracy improved from 79.82% to 80.15%



    Epoch 39/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 39/40 - Train loss: 1.4018, Val loss: 0.7232, Val acc: 80.02%



    Epoch 40/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 40/40 - Train loss: 1.3824, Val loss: 0.7208, Val acc: 80.11%


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
# plot_training_curves(training_stats)

# Evaluate the best model
accuracy, all_preds, all_targets = evaluate_best_model(
    baseline_model, val_loader, device, './checkpoints/convnext_baseline/best.pth'
)
```


    Evaluating:   0%|          | 0/157 [00:00<?, ?it/s]


    Test accuracy: 80.15%


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



    Epoch 1/10 [ResNet]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation [ResNet]:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 1/10 - ResNet - Train loss: 1.4578, Val loss: 1.1520, Val acc: 58.65%



    Epoch 2/10 [ResNet]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation [ResNet]:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 2/10 - ResNet - Train loss: 1.0816, Val loss: 1.0342, Val acc: 64.84%



    Epoch 3/10 [ResNet]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation [ResNet]:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 3/10 - ResNet - Train loss: 0.9271, Val loss: 0.9233, Val acc: 67.84%



    Epoch 4/10 [ResNet]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation [ResNet]:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 4/10 - ResNet - Train loss: 0.8238, Val loss: 0.8093, Val acc: 71.99%



    Epoch 5/10 [ResNet]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation [ResNet]:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 5/10 - ResNet - Train loss: 0.7514, Val loss: 0.7745, Val acc: 73.53%



    Epoch 6/10 [ResNet]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation [ResNet]:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 6/10 - ResNet - Train loss: 0.6985, Val loss: 0.6933, Val acc: 76.19%



    Epoch 7/10 [ResNet]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation [ResNet]:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 7/10 - ResNet - Train loss: 0.6549, Val loss: 0.6837, Val acc: 76.34%



    Epoch 8/10 [ResNet]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation [ResNet]:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 8/10 - ResNet - Train loss: 0.6125, Val loss: 0.6911, Val acc: 76.21%



    Epoch 9/10 [ResNet]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation [ResNet]:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 9/10 - ResNet - Train loss: 0.5729, Val loss: 0.6831, Val acc: 76.75%



    Epoch 10/10 [ResNet]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation [ResNet]:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 10/10 - ResNet - Train loss: 0.5417, Val loss: 0.6584, Val acc: 77.90%



    
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
    Best validation accuracy: 80.15%
    Final validation accuracy: 80.11%
    
    ConvNeXt-Tiny Architecture:
    - Total parameters: 27,827,818
    - Trainable parameters: 27,827,818
    
    Training details:
    - Epochs: 40
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


# [ADDED] Functions for running full experiments and analyzing results


```python
# [ADDED] Function to run the full experiment
def run_full_experiment():
    """Run the full experiment of training and comparing all model variants"""
    # Create model variants
    
    models_to_test = create_model_variants(num_classes=10)
    
    # Train all variants
    all_results = train_all_variants(models_to_test, train_loader, val_loader, device, epochs=30)
    
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

    
    ========================================
    Training convnext_gelu_layernorm...
    ========================================
    



    Epoch 1/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 1/30 - Train loss: 2.5377, Val loss: 2.5782, Val acc: 7.48%
    Validation accuracy improved from 0.00% to 7.48%



    Epoch 2/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 2/30 - Train loss: 2.0720, Val loss: 1.6262, Val acc: 44.19%
    Validation accuracy improved from 7.48% to 44.19%



    Epoch 3/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 3/30 - Train loss: 1.9686, Val loss: 1.4826, Val acc: 50.27%
    Validation accuracy improved from 44.19% to 50.27%



    Epoch 4/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 4/30 - Train loss: 1.9151, Val loss: 1.4110, Val acc: 52.43%
    Validation accuracy improved from 50.27% to 52.43%



    Epoch 5/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 5/30 - Train loss: 1.8915, Val loss: 1.3621, Val acc: 54.57%
    Validation accuracy improved from 52.43% to 54.57%



    Epoch 6/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 6/30 - Train loss: 1.8785, Val loss: 1.3499, Val acc: 55.34%
    Validation accuracy improved from 54.57% to 55.34%



    Epoch 7/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 7/30 - Train loss: 1.8428, Val loss: 1.3130, Val acc: 56.83%
    Validation accuracy improved from 55.34% to 56.83%



    Epoch 8/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 8/30 - Train loss: 1.7935, Val loss: 1.2138, Val acc: 60.77%
    Validation accuracy improved from 56.83% to 60.77%



    Epoch 9/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 9/30 - Train loss: 1.7799, Val loss: 1.1544, Val acc: 62.52%
    Validation accuracy improved from 60.77% to 62.52%



    Epoch 10/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 10/30 - Train loss: 1.7635, Val loss: 1.2309, Val acc: 61.37%



    Epoch 11/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 11/30 - Train loss: 1.7145, Val loss: 1.1003, Val acc: 65.02%
    Validation accuracy improved from 62.52% to 65.02%



    Epoch 12/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 12/30 - Train loss: 1.7036, Val loss: 1.0058, Val acc: 67.83%
    Validation accuracy improved from 65.02% to 67.83%



    Epoch 13/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 13/30 - Train loss: 1.7122, Val loss: 1.0740, Val acc: 67.32%



    Epoch 14/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 14/30 - Train loss: 1.6619, Val loss: 1.0316, Val acc: 68.55%
    Validation accuracy improved from 67.83% to 68.55%



    Epoch 15/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 15/30 - Train loss: 1.6540, Val loss: 1.0857, Val acc: 68.36%



    Epoch 16/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 16/30 - Train loss: 1.6268, Val loss: 0.9614, Val acc: 70.82%
    Validation accuracy improved from 68.55% to 70.82%



    Epoch 17/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 17/30 - Train loss: 1.6078, Val loss: 0.9521, Val acc: 71.57%
    Validation accuracy improved from 70.82% to 71.57%



    Epoch 18/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 18/30 - Train loss: 1.6040, Val loss: 0.8928, Val acc: 73.00%
    Validation accuracy improved from 71.57% to 73.00%



    Epoch 19/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 19/30 - Train loss: 1.5867, Val loss: 0.9483, Val acc: 71.77%



    Epoch 20/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 20/30 - Train loss: 1.5656, Val loss: 0.9049, Val acc: 73.39%
    Validation accuracy improved from 73.00% to 73.39%



    Epoch 21/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 21/30 - Train loss: 1.5536, Val loss: 0.8480, Val acc: 74.76%
    Validation accuracy improved from 73.39% to 74.76%



    Epoch 22/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 22/30 - Train loss: 1.5372, Val loss: 0.8482, Val acc: 75.13%
    Validation accuracy improved from 74.76% to 75.13%



    Epoch 23/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 23/30 - Train loss: 1.4965, Val loss: 0.8450, Val acc: 76.06%
    Validation accuracy improved from 75.13% to 76.06%



    Epoch 24/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 24/30 - Train loss: 1.4982, Val loss: 0.8108, Val acc: 76.12%
    Validation accuracy improved from 76.06% to 76.12%



    Epoch 25/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 25/30 - Train loss: 1.4844, Val loss: 0.8139, Val acc: 76.28%
    Validation accuracy improved from 76.12% to 76.28%



    Epoch 26/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 26/30 - Train loss: 1.5045, Val loss: 0.8209, Val acc: 76.54%
    Validation accuracy improved from 76.28% to 76.54%



    Epoch 27/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 27/30 - Train loss: 1.5152, Val loss: 0.8059, Val acc: 76.91%
    Validation accuracy improved from 76.54% to 76.91%



    Epoch 28/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 28/30 - Train loss: 1.4705, Val loss: 0.8005, Val acc: 76.87%



    Epoch 29/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 29/30 - Train loss: 1.4756, Val loss: 0.7890, Val acc: 77.16%
    Validation accuracy improved from 76.91% to 77.16%



    Epoch 30/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 30/30 - Train loss: 1.4591, Val loss: 0.7888, Val acc: 77.32%
    Validation accuracy improved from 77.16% to 77.32%
    
    ========================================
    Training convnext_silu_layernorm...
    ========================================
    



    Epoch 1/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 1/30 - Train loss: 2.4327, Val loss: 2.4213, Val acc: 11.56%
    Validation accuracy improved from 0.00% to 11.56%



    Epoch 2/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 2/30 - Train loss: 2.0801, Val loss: 1.6351, Val acc: 43.37%
    Validation accuracy improved from 11.56% to 43.37%



    Epoch 3/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 3/30 - Train loss: 1.9879, Val loss: 1.5294, Val acc: 48.94%
    Validation accuracy improved from 43.37% to 48.94%



    Epoch 4/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 4/30 - Train loss: 1.9399, Val loss: 1.4823, Val acc: 49.70%
    Validation accuracy improved from 48.94% to 49.70%



    Epoch 5/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 5/30 - Train loss: 1.9108, Val loss: 1.4142, Val acc: 52.70%
    Validation accuracy improved from 49.70% to 52.70%



    Epoch 6/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 6/30 - Train loss: 1.8926, Val loss: 1.3789, Val acc: 54.19%
    Validation accuracy improved from 52.70% to 54.19%



    Epoch 7/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 7/30 - Train loss: 1.8438, Val loss: 1.3081, Val acc: 57.18%
    Validation accuracy improved from 54.19% to 57.18%



    Epoch 8/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 8/30 - Train loss: 1.8220, Val loss: 1.2329, Val acc: 59.06%
    Validation accuracy improved from 57.18% to 59.06%



    Epoch 9/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 9/30 - Train loss: 1.8112, Val loss: 1.2582, Val acc: 59.40%
    Validation accuracy improved from 59.06% to 59.40%



    Epoch 10/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 10/30 - Train loss: 1.7728, Val loss: 1.1645, Val acc: 62.48%
    Validation accuracy improved from 59.40% to 62.48%



    Epoch 11/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 11/30 - Train loss: 1.7525, Val loss: 1.1058, Val acc: 64.59%
    Validation accuracy improved from 62.48% to 64.59%



    Epoch 12/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 12/30 - Train loss: 1.7321, Val loss: 1.1311, Val acc: 64.82%
    Validation accuracy improved from 64.59% to 64.82%



    Epoch 13/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 13/30 - Train loss: 1.7117, Val loss: 1.1688, Val acc: 66.00%
    Validation accuracy improved from 64.82% to 66.00%



    Epoch 14/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 14/30 - Train loss: 1.6815, Val loss: 1.0569, Val acc: 68.15%
    Validation accuracy improved from 66.00% to 68.15%



    Epoch 15/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 15/30 - Train loss: 1.6534, Val loss: 0.9927, Val acc: 70.61%
    Validation accuracy improved from 68.15% to 70.61%



    Epoch 16/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 16/30 - Train loss: 1.6583, Val loss: 0.9998, Val acc: 70.40%



    Epoch 17/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 17/30 - Train loss: 1.6215, Val loss: 0.9586, Val acc: 71.55%
    Validation accuracy improved from 70.61% to 71.55%



    Epoch 18/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 18/30 - Train loss: 1.5976, Val loss: 0.9153, Val acc: 72.68%
    Validation accuracy improved from 71.55% to 72.68%



    Epoch 19/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 19/30 - Train loss: 1.5950, Val loss: 0.8959, Val acc: 73.12%
    Validation accuracy improved from 72.68% to 73.12%



    Epoch 20/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 20/30 - Train loss: 1.5737, Val loss: 0.8840, Val acc: 73.80%
    Validation accuracy improved from 73.12% to 73.80%



    Epoch 21/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 21/30 - Train loss: 1.5585, Val loss: 0.8615, Val acc: 73.96%
    Validation accuracy improved from 73.80% to 73.96%



    Epoch 22/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 22/30 - Train loss: 1.5444, Val loss: 0.8265, Val acc: 75.35%
    Validation accuracy improved from 73.96% to 75.35%



    Epoch 23/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 23/30 - Train loss: 1.5154, Val loss: 0.8241, Val acc: 75.60%
    Validation accuracy improved from 75.35% to 75.60%



    Epoch 24/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 24/30 - Train loss: 1.5243, Val loss: 0.8091, Val acc: 76.67%
    Validation accuracy improved from 75.60% to 76.67%



    Epoch 25/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 25/30 - Train loss: 1.5135, Val loss: 0.8084, Val acc: 76.72%
    Validation accuracy improved from 76.67% to 76.72%



    Epoch 26/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 26/30 - Train loss: 1.4911, Val loss: 0.8176, Val acc: 76.80%
    Validation accuracy improved from 76.72% to 76.80%



    Epoch 27/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 27/30 - Train loss: 1.4632, Val loss: 0.7962, Val acc: 77.27%
    Validation accuracy improved from 76.80% to 77.27%



    Epoch 28/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 28/30 - Train loss: 1.4876, Val loss: 0.7869, Val acc: 77.44%
    Validation accuracy improved from 77.27% to 77.44%



    Epoch 29/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 29/30 - Train loss: 1.5046, Val loss: 0.7862, Val acc: 77.18%



    Epoch 30/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 30/30 - Train loss: 1.4799, Val loss: 0.7863, Val acc: 77.41%
    
    ========================================
    Training convnext_mish_layernorm...
    ========================================
    



    Epoch 1/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 1/30 - Train loss: 2.4855, Val loss: 2.4931, Val acc: 8.70%
    Validation accuracy improved from 0.00% to 8.70%



    Epoch 2/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 2/30 - Train loss: 2.0889, Val loss: 1.6797, Val acc: 44.27%
    Validation accuracy improved from 8.70% to 44.27%



    Epoch 3/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 3/30 - Train loss: 1.9869, Val loss: 1.5788, Val acc: 47.74%
    Validation accuracy improved from 44.27% to 47.74%



    Epoch 4/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 4/30 - Train loss: 1.9414, Val loss: 1.5037, Val acc: 50.24%
    Validation accuracy improved from 47.74% to 50.24%



    Epoch 5/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 5/30 - Train loss: 1.9284, Val loss: 1.4766, Val acc: 52.06%
    Validation accuracy improved from 50.24% to 52.06%



    Epoch 6/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 6/30 - Train loss: 1.8995, Val loss: 1.4140, Val acc: 54.04%
    Validation accuracy improved from 52.06% to 54.04%



    Epoch 7/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 7/30 - Train loss: 1.8693, Val loss: 1.3979, Val acc: 54.54%
    Validation accuracy improved from 54.04% to 54.54%



    Epoch 8/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 8/30 - Train loss: 1.8434, Val loss: 1.2673, Val acc: 58.44%
    Validation accuracy improved from 54.54% to 58.44%



    Epoch 9/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 9/30 - Train loss: 1.8274, Val loss: 1.2712, Val acc: 59.61%
    Validation accuracy improved from 58.44% to 59.61%



    Epoch 10/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 10/30 - Train loss: 1.7951, Val loss: 1.2330, Val acc: 60.48%
    Validation accuracy improved from 59.61% to 60.48%



    Epoch 11/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 11/30 - Train loss: 1.7664, Val loss: 1.1540, Val acc: 63.73%
    Validation accuracy improved from 60.48% to 63.73%



    Epoch 12/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 12/30 - Train loss: 1.7246, Val loss: 1.0651, Val acc: 65.76%
    Validation accuracy improved from 63.73% to 65.76%



    Epoch 13/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 13/30 - Train loss: 1.7100, Val loss: 1.0596, Val acc: 66.92%
    Validation accuracy improved from 65.76% to 66.92%



    Epoch 14/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 14/30 - Train loss: 1.6886, Val loss: 1.0793, Val acc: 67.73%
    Validation accuracy improved from 66.92% to 67.73%



    Epoch 15/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 15/30 - Train loss: 1.6748, Val loss: 1.0181, Val acc: 69.20%
    Validation accuracy improved from 67.73% to 69.20%



    Epoch 16/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 16/30 - Train loss: 1.6544, Val loss: 0.9828, Val acc: 70.03%
    Validation accuracy improved from 69.20% to 70.03%



    Epoch 17/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 17/30 - Train loss: 1.6222, Val loss: 0.9674, Val acc: 71.71%
    Validation accuracy improved from 70.03% to 71.71%



    Epoch 18/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 18/30 - Train loss: 1.6161, Val loss: 0.9443, Val acc: 72.31%
    Validation accuracy improved from 71.71% to 72.31%



    Epoch 19/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 19/30 - Train loss: 1.5877, Val loss: 0.8928, Val acc: 73.34%
    Validation accuracy improved from 72.31% to 73.34%



    Epoch 20/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 20/30 - Train loss: 1.5826, Val loss: 0.8936, Val acc: 73.79%
    Validation accuracy improved from 73.34% to 73.79%



    Epoch 21/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 21/30 - Train loss: 1.5580, Val loss: 0.8666, Val acc: 74.90%
    Validation accuracy improved from 73.79% to 74.90%



    Epoch 22/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 22/30 - Train loss: 1.5507, Val loss: 0.8497, Val acc: 75.15%
    Validation accuracy improved from 74.90% to 75.15%



    Epoch 23/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 23/30 - Train loss: 1.5167, Val loss: 0.8085, Val acc: 76.09%
    Validation accuracy improved from 75.15% to 76.09%



    Epoch 24/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 24/30 - Train loss: 1.5244, Val loss: 0.8355, Val acc: 75.68%



    Epoch 25/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 25/30 - Train loss: 1.5232, Val loss: 0.8196, Val acc: 76.50%
    Validation accuracy improved from 76.09% to 76.50%



    Epoch 26/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 26/30 - Train loss: 1.5007, Val loss: 0.8101, Val acc: 76.88%
    Validation accuracy improved from 76.50% to 76.88%



    Epoch 27/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 27/30 - Train loss: 1.4959, Val loss: 0.7892, Val acc: 77.22%
    Validation accuracy improved from 76.88% to 77.22%



    Epoch 28/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 28/30 - Train loss: 1.4810, Val loss: 0.7835, Val acc: 77.21%



    Epoch 29/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 29/30 - Train loss: 1.4759, Val loss: 0.7845, Val acc: 77.35%
    Validation accuracy improved from 77.22% to 77.35%



    Epoch 30/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 30/30 - Train loss: 1.4686, Val loss: 0.7826, Val acc: 77.31%
    
    ========================================
    Training convnext_hardswish_layernorm...
    ========================================
    



    Epoch 1/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 1/30 - Train loss: 2.4665, Val loss: 2.4705, Val acc: 8.62%
    Validation accuracy improved from 0.00% to 8.62%



    Epoch 2/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 2/30 - Train loss: 2.0721, Val loss: 1.6541, Val acc: 43.16%
    Validation accuracy improved from 8.62% to 43.16%



    Epoch 3/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 3/30 - Train loss: 1.9863, Val loss: 1.5769, Val acc: 47.47%
    Validation accuracy improved from 43.16% to 47.47%



    Epoch 4/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 4/30 - Train loss: 1.9537, Val loss: 1.5015, Val acc: 49.76%
    Validation accuracy improved from 47.47% to 49.76%



    Epoch 5/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 5/30 - Train loss: 1.9105, Val loss: 1.4535, Val acc: 51.61%
    Validation accuracy improved from 49.76% to 51.61%



    Epoch 6/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 6/30 - Train loss: 1.8975, Val loss: 1.4360, Val acc: 52.68%
    Validation accuracy improved from 51.61% to 52.68%



    Epoch 7/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 7/30 - Train loss: 1.8589, Val loss: 1.3302, Val acc: 57.03%
    Validation accuracy improved from 52.68% to 57.03%



    Epoch 8/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 8/30 - Train loss: 1.8248, Val loss: 1.2248, Val acc: 59.38%
    Validation accuracy improved from 57.03% to 59.38%



    Epoch 9/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 9/30 - Train loss: 1.8142, Val loss: 1.2009, Val acc: 61.51%
    Validation accuracy improved from 59.38% to 61.51%



    Epoch 10/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 10/30 - Train loss: 1.7831, Val loss: 1.1725, Val acc: 63.27%
    Validation accuracy improved from 61.51% to 63.27%



    Epoch 11/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 11/30 - Train loss: 1.7488, Val loss: 1.1218, Val acc: 62.91%



    Epoch 12/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 12/30 - Train loss: 1.7313, Val loss: 1.1706, Val acc: 65.44%
    Validation accuracy improved from 63.27% to 65.44%



    Epoch 13/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 13/30 - Train loss: 1.7167, Val loss: 1.0787, Val acc: 67.15%
    Validation accuracy improved from 65.44% to 67.15%



    Epoch 14/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 14/30 - Train loss: 1.6892, Val loss: 1.0382, Val acc: 68.21%
    Validation accuracy improved from 67.15% to 68.21%



    Epoch 15/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 15/30 - Train loss: 1.6508, Val loss: 1.0012, Val acc: 68.64%
    Validation accuracy improved from 68.21% to 68.64%



    Epoch 16/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 16/30 - Train loss: 1.6607, Val loss: 0.9720, Val acc: 70.16%
    Validation accuracy improved from 68.64% to 70.16%



    Epoch 17/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 17/30 - Train loss: 1.6262, Val loss: 0.9283, Val acc: 71.72%
    Validation accuracy improved from 70.16% to 71.72%



    Epoch 18/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 18/30 - Train loss: 1.6098, Val loss: 0.9503, Val acc: 72.24%
    Validation accuracy improved from 71.72% to 72.24%



    Epoch 19/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 19/30 - Train loss: 1.5854, Val loss: 0.9086, Val acc: 73.55%
    Validation accuracy improved from 72.24% to 73.55%



    Epoch 20/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 20/30 - Train loss: 1.5769, Val loss: 0.8898, Val acc: 73.75%
    Validation accuracy improved from 73.55% to 73.75%



    Epoch 21/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 21/30 - Train loss: 1.5533, Val loss: 0.8693, Val acc: 74.71%
    Validation accuracy improved from 73.75% to 74.71%



    Epoch 22/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 22/30 - Train loss: 1.5203, Val loss: 0.8102, Val acc: 75.44%
    Validation accuracy improved from 74.71% to 75.44%



    Epoch 23/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 23/30 - Train loss: 1.5256, Val loss: 0.8306, Val acc: 76.73%
    Validation accuracy improved from 75.44% to 76.73%



    Epoch 24/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 24/30 - Train loss: 1.5374, Val loss: 0.8167, Val acc: 76.11%



    Epoch 25/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 25/30 - Train loss: 1.5124, Val loss: 0.7933, Val acc: 76.92%
    Validation accuracy improved from 76.73% to 76.92%



    Epoch 26/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 26/30 - Train loss: 1.5013, Val loss: 0.7970, Val acc: 77.19%
    Validation accuracy improved from 76.92% to 77.19%



    Epoch 27/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 27/30 - Train loss: 1.4959, Val loss: 0.7911, Val acc: 77.26%
    Validation accuracy improved from 77.19% to 77.26%



    Epoch 28/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 28/30 - Train loss: 1.5162, Val loss: 0.7906, Val acc: 77.51%
    Validation accuracy improved from 77.26% to 77.51%



    Epoch 29/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 29/30 - Train loss: 1.5055, Val loss: 0.7884, Val acc: 77.54%
    Validation accuracy improved from 77.51% to 77.54%



    Epoch 30/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 30/30 - Train loss: 1.4843, Val loss: 0.7845, Val acc: 77.80%
    Validation accuracy improved from 77.54% to 77.80%
    
    ========================================
    Training convnext_gelu_groupnorm...
    ========================================
    



    Epoch 1/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 1/30 - Train loss: 2.4615, Val loss: 2.4753, Val acc: 9.03%
    Validation accuracy improved from 0.00% to 9.03%



    Epoch 2/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 2/30 - Train loss: 2.0749, Val loss: 1.6382, Val acc: 43.73%
    Validation accuracy improved from 9.03% to 43.73%



    Epoch 3/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 3/30 - Train loss: 1.9698, Val loss: 1.4909, Val acc: 50.50%
    Validation accuracy improved from 43.73% to 50.50%



    Epoch 4/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 4/30 - Train loss: 1.9046, Val loss: 1.3880, Val acc: 54.51%
    Validation accuracy improved from 50.50% to 54.51%



    Epoch 5/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 5/30 - Train loss: 1.8732, Val loss: 1.3696, Val acc: 55.60%
    Validation accuracy improved from 54.51% to 55.60%



    Epoch 6/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 6/30 - Train loss: 1.8505, Val loss: 1.3287, Val acc: 57.96%
    Validation accuracy improved from 55.60% to 57.96%



    Epoch 7/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 7/30 - Train loss: 1.8288, Val loss: 1.2827, Val acc: 61.12%
    Validation accuracy improved from 57.96% to 61.12%



    Epoch 8/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 8/30 - Train loss: 1.7728, Val loss: 1.1680, Val acc: 63.94%
    Validation accuracy improved from 61.12% to 63.94%



    Epoch 9/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 9/30 - Train loss: 1.7530, Val loss: 1.1092, Val acc: 66.12%
    Validation accuracy improved from 63.94% to 66.12%



    Epoch 10/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 10/30 - Train loss: 1.7293, Val loss: 1.0611, Val acc: 67.25%
    Validation accuracy improved from 66.12% to 67.25%



    Epoch 11/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 11/30 - Train loss: 1.6999, Val loss: 1.0525, Val acc: 68.75%
    Validation accuracy improved from 67.25% to 68.75%



    Epoch 12/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 12/30 - Train loss: 1.6792, Val loss: 0.9913, Val acc: 70.10%
    Validation accuracy improved from 68.75% to 70.10%



    Epoch 13/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 13/30 - Train loss: 1.6508, Val loss: 1.0722, Val acc: 71.52%
    Validation accuracy improved from 70.10% to 71.52%



    Epoch 14/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 14/30 - Train loss: 1.6414, Val loss: 0.9456, Val acc: 71.88%
    Validation accuracy improved from 71.52% to 71.88%



    Epoch 15/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 15/30 - Train loss: 1.6237, Val loss: 0.8949, Val acc: 73.31%
    Validation accuracy improved from 71.88% to 73.31%



    Epoch 16/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 16/30 - Train loss: 1.6036, Val loss: 0.8802, Val acc: 74.74%
    Validation accuracy improved from 73.31% to 74.74%



    Epoch 17/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 17/30 - Train loss: 1.5689, Val loss: 0.8519, Val acc: 75.58%
    Validation accuracy improved from 74.74% to 75.58%



    Epoch 18/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 18/30 - Train loss: 1.5470, Val loss: 0.8420, Val acc: 75.69%
    Validation accuracy improved from 75.58% to 75.69%



    Epoch 19/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 19/30 - Train loss: 1.5281, Val loss: 0.8101, Val acc: 76.82%
    Validation accuracy improved from 75.69% to 76.82%



    Epoch 20/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 20/30 - Train loss: 1.5344, Val loss: 0.8079, Val acc: 77.84%
    Validation accuracy improved from 76.82% to 77.84%



    Epoch 21/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 21/30 - Train loss: 1.5110, Val loss: 0.7750, Val acc: 77.85%
    Validation accuracy improved from 77.84% to 77.85%



    Epoch 22/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 22/30 - Train loss: 1.4835, Val loss: 0.7570, Val acc: 78.59%
    Validation accuracy improved from 77.85% to 78.59%



    Epoch 23/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 23/30 - Train loss: 1.5032, Val loss: 0.7546, Val acc: 79.32%
    Validation accuracy improved from 78.59% to 79.32%



    Epoch 24/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 24/30 - Train loss: 1.4718, Val loss: 0.7291, Val acc: 79.95%
    Validation accuracy improved from 79.32% to 79.95%



    Epoch 25/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 25/30 - Train loss: 1.4408, Val loss: 0.7168, Val acc: 79.71%



    Epoch 26/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 26/30 - Train loss: 1.4354, Val loss: 0.7292, Val acc: 79.97%
    Validation accuracy improved from 79.95% to 79.97%



    Epoch 27/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 27/30 - Train loss: 1.4143, Val loss: 0.7206, Val acc: 80.21%
    Validation accuracy improved from 79.97% to 80.21%



    Epoch 28/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 28/30 - Train loss: 1.4294, Val loss: 0.7092, Val acc: 80.26%
    Validation accuracy improved from 80.21% to 80.26%



    Epoch 29/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 29/30 - Train loss: 1.4153, Val loss: 0.7094, Val acc: 80.80%
    Validation accuracy improved from 80.26% to 80.80%



    Epoch 30/40 [Train]:   0%|          | 0/781 [00:00<?, ?it/s]



    Validation:   0%|          | 0/157 [00:00<?, ?it/s]


    Epoch 30/30 - Train loss: 1.4107, Val loss: 0.7053, Val acc: 80.82%
    Validation accuracy improved from 80.80% to 80.82%



    
![png](output_38_451.png)
    


                              Model Best Validation Accuracy Training Accuracy  \
    4       convnext_gelu_groupnorm                   80.82%             0.00%   
    3  convnext_hardswish_layernorm                   77.80%             0.00%   
    1       convnext_silu_layernorm                   77.44%             0.00%   
    2       convnext_mish_layernorm                   77.35%             0.00%   
    0       convnext_gelu_layernorm                   77.32%             0.00%   
    
      Validation Loss  Best Epoch  
    4          0.7053          30  
    3          0.7845          30  
    1          0.7869          28  
    2          0.7845          29  
    0          0.7888          30  



    
![png](output_38_453.png)
    


The notebook now includes comprehensive improvements to the ConvNeXt architecture, focusing on activation functions and normalization strategies. The implementation allows for systematic comparison of different variants to determine which combinations yield the best performance on image classification tasks.
