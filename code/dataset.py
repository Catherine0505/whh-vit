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
from tqdm import tqdm
import pathlib



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


class ChildSubjectGroupDataset(torch.utils.data.Dataset): 
    def __init__(self, root_folder, type="train", view=90): 
        super().__init__()

        np.random.seed(297)

        self.root_folder = root_folder 
        self.root_path = pathlib.Path(self.root_folder)

        subjects = []
        images = []
        labels = []  

        # Load all subjects, images and labels 
        image_files = list(self.root_path.glob("**/scene*_*.png"))
        for image_file in tqdm(image_files): 
            subject = re.findall(r'scene(\d+)_\d{3}.png', image_file.name)[0]
            image = iio.imread(image_file) / 255.

            label_file = image_file.with_suffix(".txt")
            lines = open(label_file, "r").readlines()
            assert len(lines) == 1

            label = float(re.findall(r".*: ([0-9\.]*)", lines[0])[0]) 

            subjects.append(subject)
            images.append(image)
            labels.append(label)
        
        # Train-val-test split by 80-10-10 on subjects 
        unique_subjects = np.unique(np.array(subjects))
        num_subjects = len(unique_subjects)

        num_train = int(num_subjects * 0.8)
        num_val = int((num_subjects - num_train) * 0.5) 
        num_test = num_subjects - num_train - num_val 

        np.random.shuffle(unique_subjects)

        train_subjects = unique_subjects[:num_train]
        val_subjects = unique_subjects[num_train:-num_test]
        test_subjects = unique_subjects[-num_test:]
        
        # Determine actual subjects used based on dataset type: train, val or test
        if type == "train": 
            self.subjects = train_subjects
        elif type == "val":
            self.subjects = val_subjects
        elif type == "test": 
            self.subjects = test_subjects 

        # Load in all views and corresponding labels of a subject each time. One element represents 
        # one subject. Images stacked in the 0-th dimension. 
        self.images = []
        self.labels = []

        images = np.array(images)
        labels = np.array(labels)
        subjects = np.array(subjects)

        for subject in self.subjects: 
            self.images.append(images[np.where(subjects == subject)])
            self.labels.append(labels[np.where(subjects == subject)])

        
    def __len__(self): 
        assert len(self.images) == len(self.subjects)
        assert len(self.labels) == len(self.subjects)
        return len(self.subjects)


    def __getitem__(self, idx):
        result = {}
        result["images"] = einops.rearrange(torch.from_numpy(self.images[idx]), 'n h w c -> n c h w')
        result["labels"] = torch.from_numpy(self.labels[idx]).unsqueeze(-1)

        return result
