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


class ChildImageDataset(torch.utils.data.Dataset): 
    def __init__(self, root_folder, type="train"): 
        super().__init__()
        np.random.seed(297)
        self.root_folder = root_folder 
        img_names = \
            sorted([os.path.splitext(f)[0] for f in os.listdir(self.root_folder) if "scene" in f and ".png" in f])
        self.valid_img_names = []
        for img_name in img_names: 
            if img_name + ".txt" in os.listdir(root_folder): 
                self.valid_img_names.append(img_name)
        self.valid_img_names = sorted(self.valid_img_names)
        np.random.shuffle(self.valid_img_names)
        train_len = int(0.8 * len(self.valid_img_names))
        val_len = (len(self.valid_img_names) - train_len) // 2

        if type == "train":
            self.img_names = self.valid_img_names[:train_len]
        elif type == "val": 
            self.img_names = self.valid_img_names[train_len: train_len + val_len]
        elif type == "test": 
            self.img_names = self.valid_img_names[train_len + val_len:]
        
    def __len__(self): 
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        x = iio.imread(os.path.join(self.root_folder, self.img_names[idx] + ".png")) / 255.
        gt_file = open(os.path.join(self.root_folder, self.img_names[idx] + ".txt"), "r")
        lines = gt_file.readlines()
        assert len(lines) == 1
        y = [float(re.findall(r".*: ([0-9\.]*)", lines[0])[0])]
        x, y = torch.Tensor(x), torch.Tensor(y)
        x = einops.rearrange(x, 'h w c -> c h w')
        return x, y


class ChildSubjectDataset(torch.utils.data.Dataset): 
    def __init__(self, root_folder, type="train", view=90): 
        super().__init__()
        np.random.seed(297)
        self.root_folder = root_folder 
        self.view = str(view).rjust(3, "0")

        img_names = \
            sorted([os.path.splitext(f)[0] for f in os.listdir(self.root_folder) if "scene" in f and ".png" in f])
        self.valid_img_names = []
        for img_name in img_names: 
            if img_name + ".txt" in os.listdir(root_folder): 
                self.valid_img_names.append(img_name)

        self.valid_subject_names = list(set([n.split("_")[0] for n in self.valid_img_names]))
        self.valid_subject_names = sorted(self.valid_subject_names)
        
        np.random.shuffle(self.valid_subject_names)
        train_len = int(0.8 * len(self.valid_subject_names))
        val_len = (len(self.valid_subject_names) - train_len) // 2

        if type == "train":
            self.subject_names = self.valid_subject_names[:train_len]
        elif type == "val": 
            self.subject_names = self.valid_subject_names[train_len: train_len + val_len]
        elif type == "test": 
            self.subject_names = self.valid_subject_names[train_len + val_len:]
        
    def __len__(self): 
        return len(self.subject_names)

    def __getitem__(self, idx):
        subject_name = self.subject_names[idx]
        img_name = "_".join([subject_name, self.view])
        x = iio.imread(os.path.join(self.root_folder, img_name + ".png")) / 255.
        gt_file = open(os.path.join(self.root_folder, img_name + ".txt"), "r")
        lines = gt_file.readlines()
        assert len(lines) == 1
        y = [float(re.findall(r".*: ([0-9\.]*)", lines[0])[0])]
        x, y = torch.Tensor(x), torch.Tensor(y)
        x = einops.rearrange(x, 'h w c -> c h w')
        return x, y


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

dataset_test = ChildSubjectDataset(root_folder="/homes/1/sg1526/misc/data", type="test")
checkpoint_file = "/homes/1/sg1526/misc/trained_weights/trainbysubject_lr3e-05/googlevitbase_patch16-224_linearhead_best_362_loss13.506065766016642.pth"
validation(dataset_test=dataset_test, checkpoint_file=checkpoint_file, train_by="subject")

dataset_test = ChildImageDataset(root_folder="/homes/1/sg1526/misc/data", type="test")
checkpoint_file = "/homes/1/sg1526/misc/trained_weights/trainbyimage/googlevitbase_patch16-224_linearhead_best_154_loss6.484924050477835.pth"
validation(dataset_test=dataset_test, checkpoint_file=checkpoint_file, train_by="image")