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

import dataset
from linearhead import LinearHead


######################################
#   Train with images 
######################################

def train(dataset_train, dataset_val, train_by):
    print(dataset_train[0][0].shape)

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=8, shuffle=True)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=8, shuffle=True)

    configuration = ViTConfig(image_size=224, patch_size=16)
    model = ViTModel.from_pretrained('google/vit-base-patch16-224', add_pooling_layer=False,
                                    cache_dir="/autofs/space/saffron_001/users/catherine_gai/misc/model_weights")
    linear_head = LinearHead(input_dim=768, hidden_features=[256, 128, 64, 32, 8], output_dim=1)
    preprocessor = ViTImageProcessor(do_resize=True, size={"height": 224, "width": 224}, 
                                    do_rescale=False, do_normalize=False)

    lr = 0.00003
    optimizer = optim.Adam(linear_head.parameters(), lr=lr)
    num_iters = 1000

    best_validation_loss = np.inf
    linear_weight_save_folder = f"/autofs/space/saffron_001/users/catherine_gai/misc/trained_weights/trainby{train_by}_lr{lr}"
    training_curve_save_folder = f"/autofs/space/saffron_001/users/catherine_gai/misc/training_curve/trainby{train_by}_lr{lr}"
    if not os.path.exists(linear_weight_save_folder):
        os.makedirs(linear_weight_save_folder, exist_ok=False)
    if not os.path.exists(training_curve_save_folder): 
        os.makedirs(training_curve_save_folder, exist_ok=False)
    linear_weight_save_name = "googlevitbase_patch16-224_linearhead"

    training_losses = [] 
    validation_losses = []
    for iter in range(num_iters):
        print(f"Iter {iter} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        training_losses_it = []
        for batch in dataloader_train: 
            x, y = batch 
            x = preprocessor.preprocess(x, return_tensors="pt", 
                data_format="channels_first", input_data_format="channels_first")
            
            optimizer.zero_grad()
            with torch.no_grad():
                outputs = model(**x)
                # print(outputs.last_hidden_state.shape)
            outputs = linear_head(outputs.last_hidden_state[:, 0])
            # print(outputs.shape)
            # print(outputs)

            loss = ((y - outputs) ** 2).mean()
            training_losses_it.append(loss.item())
            loss.backward()
            optimizer.step()
        print("Training loss:", np.array(training_losses_it).mean())
        training_losses.append(np.array(training_losses_it).mean())

        linear_head.eval()
        validation_losses_it = []
        for batch in dataloader_val: 
            x, y = batch 
            x = preprocessor.preprocess(x, return_tensors="pt", 
                data_format="channels_first", input_data_format="channels_first")
            with torch.no_grad():
                outputs = model(**x)
            outputs = linear_head(outputs.last_hidden_state[:, 0])
            loss = ((y - outputs) ** 2).mean()
            validation_losses_it.append(loss.item())
        print("Validation loss:", np.array(validation_losses_it).mean())
        if np.array(validation_losses_it).mean() < best_validation_loss: 
            best_validation_loss = np.array(validation_losses_it).mean()
            torch.save(linear_head.state_dict(), 
                os.path.join(linear_weight_save_folder, linear_weight_save_name + f"_best_{iter}_loss{best_validation_loss}.pth"))
        validation_losses.append(np.array(validation_losses_it).mean())
        linear_head.train()

        if iter > 0 and iter % 100 == 0: 
            plt.plot(np.arange(iter + 1), training_losses, label = "Training Loss")
            plt.plot(np.arange(iter + 1), validation_losses, label = "Validation Loss")
            plt.legend()
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.title("Training Curve")
            plt.savefig(os.path.join(training_curve_save_folder, linear_weight_save_name + f"_iter{iter}.png"))
            plt.close()



#####################################################################################
#   Collective model with multi-view. No fusion with image feature, Train by subject.
#####################################################################################

def train_collective_nofusion(dataset_train, dataset_val, train_bs, val_bs, save_weight_folder):
    # print(dataset_train[0][0].shape)

    save_weight_path = pathlib.Path(save_weight_folder)

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=train_bs, shuffle=True)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=val_bs, shuffle=True)

    configuration = ViTConfig(image_size=224, patch_size=16)
    pretrainedvit_save_path = save_weight_path / "pretrainedvit_weights"
    model = ViTModel.from_pretrained('google/vit-base-patch16-224', add_pooling_layer=False,
                                    cache_dir=pretrainedvit_save_path)
    linear_head = LinearHead(input_dim=768, hidden_features=[256, 128, 64, 32, 8], output_dim=1)
    preprocessor = ViTImageProcessor(do_resize=True, size={"height": 224, "width": 224}, 
                                    do_rescale=False, do_normalize=False)
    model = model.cuda()
    linear_head = linear_head.cuda()

    lr = 0.001
    optimizer = optim.Adam(linear_head.parameters(), lr=lr)
    num_iters = 1000

    # Construct path to save trained linear head weights
    path_identifier = [f for f in os.listdir(pretrainedvit_save_path) \
                       if os.path.isdir(pretrainedvit_save_path / f) and f[0] != "."]
    print(path_identifier)
    assert len(path_identifier) == 1 
    path_identifier = path_identifier[0]
    trainedlinearhead_save_path = \
        pathlib.Path(save_weight_path / "trainedlinearhead_weights" / path_identifier / "collective_nofusion" / f"lr{lr}")
    trainingcurve_save_path = \
        pathlib.Path(save_weight_path / "training_curve" / path_identifier / "collective_nofusion" / f"lr{lr}")
    
    if not os.path.exists(trainedlinearhead_save_path):
        os.makedirs(trainedlinearhead_save_path, exist_ok=False)
    if not os.path.exists(trainingcurve_save_path): 
        os.makedirs(trainingcurve_save_path, exist_ok=False)

    training_losses = [] 
    validation_losses = []
    best_validation_loss = np.inf
    for iter in range(num_iters):
        print(f"Iter {iter} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        training_losses_it = []
        for batch in dataloader_train: 
            x, y = batch["images"], batch["labels"]

            n = x.shape[1]
            x = einops.rearrange(x, "b n c h w -> (b n) c h w")
            y = y.mean(dim=1)

            x = preprocessor.preprocess(x, return_tensors="pt", 
                data_format="channels_first", input_data_format="channels_first")
            
            x = x.to("cuda")
            y = y.cuda()
            
            optimizer.zero_grad()
            with torch.no_grad():
                outputs = model(**x)
                # print(outputs.last_hidden_state.shape)

            outputs = outputs.last_hidden_state[:, 0]
            outputs = einops.rearrange(outputs, "(b n) c -> b n c", n=n)
            outputs = outputs.mean(dim=1)
            outputs = linear_head(outputs)
            
            # print(outputs.shape)
            # print(outputs)

            loss = ((y - outputs) ** 2).mean()
            training_losses_it.append(loss.item())
            loss.backward()
            optimizer.step()
        print("Training loss:", np.array(training_losses_it).mean())
        training_losses.append(np.array(training_losses_it).mean())

        linear_head.eval()
        validation_losses_it = []
        for batch in dataloader_val: 
            x, y = batch["images"], batch["labels"]

            n = x.shape[1]
            x = einops.rearrange(x, "b n c h w -> (b n) c h w")
            y = y.mean(dim=1)

            x = preprocessor.preprocess(x, return_tensors="pt", 
                data_format="channels_first", input_data_format="channels_first")
            
            x = x.to("cuda")
            y = y.cuda()

            with torch.no_grad():
                outputs = model(**x)

            outputs = outputs.last_hidden_state[:, 0]
            outputs = einops.rearrange(outputs, "(b n) c -> b n c", n=n)
            outputs = outputs.mean(dim=1)
            outputs = linear_head(outputs)

            loss = ((y - outputs) ** 2).mean()
            validation_losses_it.append(loss.item())
        print("Validation loss:", np.array(validation_losses_it).mean())
        if np.array(validation_losses_it).mean() < best_validation_loss: 
            best_validation_loss = np.array(validation_losses_it).mean()
            torch.save(linear_head.state_dict(), 
                 trainedlinearhead_save_path / f"best_weight.pth")
        validation_losses.append(np.array(validation_losses_it).mean())
        linear_head.train()

        if iter > 0 and iter % 100 == 0: 
            plt.plot(np.arange(iter + 1), training_losses, label = "Training Loss")
            plt.plot(np.arange(iter + 1), validation_losses, label = "Validation Loss")
            plt.legend()
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.title("Training Curve")
            plt.savefig(trainingcurve_save_path / f"iter{iter}.png")
            plt.close()

    torch.save(linear_head.state_dict(), 
                 trainedlinearhead_save_path / f"last_weight.pth")
    


#####################################################################################
#   Collective model with multi-view. Append with image feature. Train by images.
#####################################################################################

def train_collective_appendfusion(dataset_train, dataset_val, train_bs, val_bs, save_weight_folder):
    # print(dataset_train[0][0].shape)

    save_weight_path = pathlib.Path(save_weight_folder)

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=train_bs, shuffle=True)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=val_bs, shuffle=True)

    configuration = ViTConfig(image_size=224, patch_size=16)
    pretrainedvit_save_path = save_weight_path / "pretrainedvit_weights"
    model = ViTModel.from_pretrained('google/vit-base-patch16-224', add_pooling_layer=False,
                                    cache_dir=pretrainedvit_save_path)
    linear_head = LinearHead(input_dim=768*2, hidden_features=[768, 256, 128, 64, 32, 8], output_dim=1)
    preprocessor = ViTImageProcessor(do_resize=True, size={"height": 224, "width": 224}, 
                                    do_rescale=False, do_normalize=False)
    model = model.cuda()
    linear_head = linear_head.cuda()

    lr = 0.0003
    optimizer = optim.Adam(linear_head.parameters(), lr=lr)
    num_iters = 1000

    # Construct path to save trained linear head weights
    path_identifier = [f for f in os.listdir(pretrainedvit_save_path) \
                       if os.path.isdir(pretrainedvit_save_path / f) and f[0] != "."]
    print(path_identifier)
    assert len(path_identifier) == 1 
    path_identifier = path_identifier[0]
    trainedlinearhead_save_path = \
        pathlib.Path(save_weight_path / "trainedlinearhead_weights" / path_identifier / "collective_appendfusion" / f"lr{lr}")
    trainingcurve_save_path = \
        pathlib.Path(save_weight_path / "training_curve" / path_identifier / "collective_appendfusion" / f"lr{lr}")
    
    if not os.path.exists(trainedlinearhead_save_path):
        os.makedirs(trainedlinearhead_save_path, exist_ok=False)
    if not os.path.exists(trainingcurve_save_path): 
        os.makedirs(trainingcurve_save_path, exist_ok=False)

    training_losses = [] 
    validation_losses = []
    best_validation_loss = np.inf
    for iter in range(num_iters):
        print(f"Iter {iter} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        training_losses_it = []
        for batch in dataloader_train: 
            x, y = batch["images"], batch["labels"]

            n = x.shape[1]
            x = einops.rearrange(x, "b n c h w -> (b n) c h w")
            y = einops.rearrange(y, "b n c -> (b n) c")

            x = preprocessor.preprocess(x, return_tensors="pt", 
                data_format="channels_first", input_data_format="channels_first")
            
            x = x.to("cuda")
            y = y.cuda()
            
            optimizer.zero_grad()
            with torch.no_grad():
                outputs = model(**x)
                # print(outputs.last_hidden_state.shape)

            outputs = outputs.last_hidden_state[:, 0]
            outputs = einops.rearrange(outputs, "(b n) c -> b n c", n=n)
            outputs_mean = outputs.mean(dim=1).unsqueeze(dim=1).repeat([1, n, 1])
            outputs = torch.concatenate([outputs, outputs_mean], dim=-1)
            outputs = einops.rearrange(outputs, "b n c -> (b n) c")
            outputs = linear_head(outputs)
            
            # print(outputs.shape)
            # print(outputs)

            loss = ((y - outputs) ** 2).mean()
            training_losses_it.append(loss.item())
            loss.backward()
            optimizer.step()
        print("Training loss:", np.array(training_losses_it).mean())
        training_losses.append(np.array(training_losses_it).mean())

        linear_head.eval()
        validation_losses_it = []
        for batch in dataloader_val: 
            x, y = batch["images"], batch["labels"]

            n = x.shape[1]
            x = einops.rearrange(x, "b n c h w -> (b n) c h w")
            y = einops.rearrange(y, "b n c -> (b n) c")

            x = preprocessor.preprocess(x, return_tensors="pt", 
                data_format="channels_first", input_data_format="channels_first")
            
            x = x.to("cuda")
            y = y.cuda()

            with torch.no_grad():
                outputs = model(**x)

            outputs = outputs.last_hidden_state[:, 0]
            outputs = einops.rearrange(outputs, "(b n) c -> b n c", n=n)
            outputs_mean = outputs.mean(dim=1).unsqueeze(dim=1).repeat([1, n, 1])
            outputs = torch.concatenate([outputs, outputs_mean], dim=-1)
            outputs = einops.rearrange(outputs, "b n c -> (b n) c")
            outputs = linear_head(outputs)

            loss = ((y - outputs) ** 2).mean()
            validation_losses_it.append(loss.item())
        print("Validation loss:", np.array(validation_losses_it).mean())
        if np.array(validation_losses_it).mean() < best_validation_loss: 
            best_validation_loss = np.array(validation_losses_it).mean()
            torch.save(linear_head.state_dict(), 
                 trainedlinearhead_save_path / f"best_weight.pth")
        validation_losses.append(np.array(validation_losses_it).mean())
        linear_head.train()

        if iter > 0 and iter % 100 == 0: 
            plt.plot(np.arange(iter + 1), training_losses, label = "Training Loss")
            plt.plot(np.arange(iter + 1), validation_losses, label = "Validation Loss")
            plt.legend()
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.title("Training Curve")
            plt.savefig(trainingcurve_save_path / f"iter{iter}.png")
            plt.close()

    torch.save(linear_head.state_dict(), 
                 trainedlinearhead_save_path / f"last_weight.pth")
        
dataset_train = dataset.ChildSubjectGroupDataset(root_folder="/homes/1/sg1526/misc/data", type="train")
dataset_val = dataset.ChildSubjectGroupDataset(root_folder="/homes/1/sg1526/misc/data", type="val")

print(len(dataset_train))
print(len(dataset_val))

print(type(dataset_train[0]))
print(dataset_train[0]["images"].shape, dataset_train[0]["labels"].shape)
print(dataset_train[0]["images"].min(), dataset_train[0]["images"].max())
train_collective_appendfusion(dataset_train=dataset_train, dataset_val=dataset_val, train_bs=4, val_bs=4, 
                 save_weight_folder="/autofs/space/celer_001/users/catherine_gai/misc/model_weights")