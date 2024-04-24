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

def validation(dataset_test, checkpoint_file, train_by): 

    # print(dataset_test[0][0].shape)

    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=8, shuffle=False)

    preprocessor = ViTImageProcessor(do_resize=True, size={"height": 224, "width": 224}, 
                                    do_rescale=False, do_normalize=False)

    model = ViTModel.from_pretrained('google/vit-base-patch16-224', add_pooling_layer=False,
                                    cache_dir="/autofs/space/saffron_001/users/catherine_gai/misc/model_weights")
    linear_head = LinearHead(input_dim=768, hidden_features=[256, 128, 64, 32, 8], output_dim=1)
    linear_head.load_state_dict(torch.load(checkpoint_file))

    predictions = []
    ground_truths = []

    for batch in dataloader_test: 
        x, y = batch 
        ground_truths.append(y.detach().cpu().numpy())

        x = preprocessor.preprocess(x, return_tensors="pt", 
            data_format="channels_first", input_data_format="channels_first")
        with torch.no_grad():
            outputs = model(**x)
        outputs = linear_head(outputs.last_hidden_state[:, 0])
        predictions.append(outputs.detach().cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    ground_truths = np.concatenate(ground_truths, axis=0)

    plt.scatter(predictions, ground_truths, label="Prediction")
    plt.plot([ground_truths.min(), ground_truths.max()], [ground_truths.min(), ground_truths.max()], 
        color = "r", label="Expected")
    plt.legend()
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.title("Performance Evaluation on Test Set")
    plt.savefig(f"/homes/1/sg1526/misc/training_curve/trainby{train_by}/validation.png")
    plt.close()

    print(f"Training by {train_by}")
    print(np.abs(predictions - ground_truths).mean())
    print((np.abs(predictions - ground_truths) <= 2).sum() / len(predictions))



########################################################################################
#   Collective model with multi-view. No fusion with image feature, Validate by subject.
########################################################################################
    
def validate_collective_nofusion(dataset_test, test_bs, save_weight_folder, eval_method="best"): 

    save_weight_path = pathlib.Path(save_weight_folder)

    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=test_bs, shuffle=False)
    
    pretrainedvit_save_path = save_weight_path / "pretrainedvit_weights"
    
    path_identifier = [f for f in os.listdir(pretrainedvit_save_path) \
                       if os.path.isdir(pretrainedvit_save_path / f) and f[0] != "."]
    assert len(path_identifier) == 1 
    path_identifier = path_identifier[0]

    trainedlinearhead_save_path = \
        pathlib.Path(save_weight_path / "trainedlinearhead_weights" / path_identifier/ "collective_nofusion_test" / f"lr0.001")
    checkpoint_file = trainedlinearhead_save_path / f"{eval_method}_weight.pth"
    

    model = ViTModel.from_pretrained('google/vit-base-patch16-224', add_pooling_layer=False,
                                    cache_dir=pretrainedvit_save_path)
    model = model.cuda()
    linear_head = LinearHead(input_dim=768, hidden_features=[256, 128, 64, 32, 8], output_dim=1)
    linear_head.load_state_dict(torch.load(checkpoint_file))
    linear_head = linear_head.cuda()
    preprocessor = ViTImageProcessor(do_resize=True, size={"height": 224, "width": 224}, 
                                    do_rescale=False, do_normalize=False)
    
    preds = np.array([])
    gts = np.array([])
    for batch in dataloader_test: 
        x, y = batch["images"].cuda(), batch["labels"].cuda()

        n = x.shape[1]
        x = einops.rearrange(x, "b n c h w -> (b n) c h w")
        y = y.mean(dim=1)
        
        x = x.cuda()
        y = y.cuda()

        with torch.no_grad():
            outputs = model(x)

        outputs = outputs.last_hidden_state[:, 0]
        outputs = einops.rearrange(outputs, "(b n) c -> b n c", n=n)
        outputs = outputs.mean(dim=1)
        outputs = linear_head(outputs)

        preds = np.concatenate([preds, outputs.flatten().detach().cpu().numpy()])
        gts = np.concatenate([gts, y.flatten().detach().cpu().numpy()])
    
    assert len(preds) == len(gts)
    print(f"Tested on {len(preds)} subjects. With {eval_method} checkpoint.")

    mse = ((gts - preds) ** 2).mean()
    mae = np.abs(gts - preds).mean()
    acc = (np.abs(gts - preds) <= 1).sum() / len(preds) * 100
    print(f"MSE: {mse}   MAE: {mae}   Accuracy within 1cm: {acc}%")

    plt.plot(gts, preds, ".")
    plt.plot(np.sort(gts), np.sort(gts), color="r")  
    plt.xlabel("Ground Truth")  
    plt.ylabel("Predictions")
    plt.title("Prediction Plot")
    plt.savefig(save_weight_path / "training_curve" / path_identifier / "collective_nofusion_test" / "lr0.001" / f"preds_{eval_method}.png")
    plt.close()




########################################################################################
#   Collective model with multi-view. Append with image feature, Validate by images.
########################################################################################
    
def validate_collective_appendfusion(dataset_test, test_bs, save_weight_folder, eval_method="best"): 

    save_weight_path = pathlib.Path(save_weight_folder)

    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=test_bs, shuffle=False)
    
    pretrainedvit_save_path = save_weight_path / "pretrainedvit_weights"
    
    path_identifier = [f for f in os.listdir(pretrainedvit_save_path) \
                       if os.path.isdir(pretrainedvit_save_path / f) and f[0] != "."]
    assert len(path_identifier) == 1 
    path_identifier = path_identifier[0]

    trainedlinearhead_save_path = \
        pathlib.Path(save_weight_path / "trainedlinearhead_weights" / path_identifier/ "collective_appendfusion" / f"lr0.0003")
    checkpoint_file = trainedlinearhead_save_path / f"{eval_method}_weight.pth"
    

    model = ViTModel.from_pretrained('google/vit-base-patch16-224', add_pooling_layer=False,
                                    cache_dir=pretrainedvit_save_path)
    model = model.cuda()
    linear_head = LinearHead(input_dim=768*2, hidden_features=[768, 256, 128, 64, 32, 8], output_dim=1)
    linear_head.load_state_dict(torch.load(checkpoint_file))
    linear_head = linear_head.cuda()
    preprocessor = ViTImageProcessor(do_resize=True, size={"height": 224, "width": 224}, 
                                    do_rescale=False, do_normalize=False)
    
    preds = np.array([])
    preds_mean = np.array([])
    gts = np.array([])
    gts_mean = np.array([])
    for batch in dataloader_test: 
        x, y = batch["images"].cuda(), batch["labels"].cuda()

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

        preds = np.concatenate([preds, outputs.flatten().detach().cpu().numpy()])
        gts = np.concatenate([gts, y.flatten().detach().cpu().numpy()])
        preds_mean = np.concatenate([preds_mean, 
            einops.rearrange(outputs, '(b n) c -> b n c', n=n).mean((-1, -2)).detach().cpu().numpy()])
        gts_mean = np.concatenate([gts_mean, 
            einops.rearrange(y, '(b n) c -> b n c', n=n).mean((-1, -2)).detach().cpu().numpy()])
    
    assert len(preds) == len(gts)
    assert len(preds_mean) == len(gts_mean)
    print(f"Tested on {len(preds)} images. With {eval_method} checkpoint.")
    print(f"Test averaged over {len(preds_mean)} subjects. With {eval_method} checkpoint.")

    mse = ((gts - preds) ** 2).mean()
    mae = np.abs(gts - preds).mean()
    acc = (np.abs(gts - preds) <= 2).sum() / len(preds) * 100

    mse_mean = ((gts_mean - preds_mean) ** 2).mean()
    mae_mean = np.abs(gts_mean - preds_mean).mean()
    acc_mean = (np.abs(gts_mean - preds_mean) <= 2).sum() / len(preds_mean) * 100

    print(f"MSE: {mse}   MAE: {mae}   Accuracy within 2cm: {acc}%")
    print(f"MSE averaged: {mse_mean}   MAE averaged: {mae_mean}   Accuracy within 2cm averaged: {acc_mean}%")

    plt.plot(gts, preds, ".")
    plt.plot(gts, gts, color="r")  
    plt.xlabel("Ground Truth")  
    plt.ylabel("Predictions")
    plt.title("Prediction Plot")
    plt.savefig(save_weight_path / "training_curve" / path_identifier / "collective_appendfusion" / "lr0.0003" / f"preds_{eval_method}.png")
    plt.close()

    plt.plot(gts_mean, preds_mean, ".")
    plt.plot(gts_mean, gts_mean, color="r")  
    plt.xlabel("Ground Truth")  
    plt.ylabel("Predictions")
    plt.title("Prediction Plot")
    plt.savefig(save_weight_path / "training_curve" / path_identifier / "collective_appendfusion" / "lr0.0003" / f"preds_{eval_method}_mean.png")
    plt.close()

    


dataset_test = dataset.ChildSubjectGroupDatasetPreload(root_folder="/home/ml_group/misc/data", type="test")
validate_collective_nofusion(dataset_test=dataset_test, test_bs=4, 
                    save_weight_folder="/home/ml_group/misc/model_weights", 
                    eval_method="last")