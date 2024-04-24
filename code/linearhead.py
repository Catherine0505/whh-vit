import transformers
from transformers import ViTConfig, ViTModel, ViTImageProcessor

import torch 
import torch.optim as optim
import matplotlib.pyplot as plt

import os
import numpy as np
import imageio.v3 as iio
import re
import einops
import tqdm
import pathlib


class LinearHead(torch.nn.Module): 
    def __init__(self, input_dim, hidden_features, output_dim): 
        super().__init__()
        self.hidden_features = hidden_features
        self.output_dim = output_dim

        self.layer_list = []

        input_dim = input_dim
        for feature in hidden_features: 
            self.layer_list.append(torch.nn.Linear(input_dim, feature))
            self.layer_list.append(torch.nn.PReLU())
            input_dim = feature 
        self.layer_list.append(torch.nn.Linear(hidden_features[-1], output_dim))

        self.layer_list = torch.nn.ModuleList(self.layer_list)
    
    def forward(self, x): 
        for layer in self.layer_list:
            x = layer(x)
        return x