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

        all_subjects = []
        self.image_paths = []
        self.labels = []  
        self.subject_imageidx_dict = {}

        # Load all subjects, images and labels 
        image_files = list(self.root_path.glob("**/*rgb*.png"))
        count = 0
        for image_file in tqdm(image_files): 
            subject = re.findall(r'child_(\d+)_rgb_\d{3}.png', image_file.name)[0]
            view = re.findall(r'child_\d+_rgb_(\d{3}).png', image_file.name)[0]
            # image = iio.imread(image_file) / 255.

            label_file = image_file.with_name(f"child_{subject}_lbl_{view}.txt")
            lines = open(label_file, "r").readlines()
            height_line = [l for l in lines if "child height" in l]
            assert len(height_line) == 1
            height_line = height_line[0]

            label = float(re.findall(r".*: ([0-9\.]*)", height_line)[0]) 

            self.image_paths.append(image_file)
            self.labels.append(label)

            if subject not in self.subject_imageidx_dict: 
                all_subjects.append(subject)
                self.subject_imageidx_dict[subject] = [count]
            else: 
                self.subject_imageidx_dict[subject].append(count)

            count += 1
        assert len(all_subjects) * len(list(self.subject_imageidx_dict.items())[0][1]) == len(self.image_paths)
        self.image_paths = np.array(self.image_paths)
        self.labels = np.array(self.labels)
        
        # Train-val-test split by 80-10-10 on subjects 
        num_subjects = len(all_subjects)

        num_train = int(num_subjects * 0.8)
        num_val = int((num_subjects - num_train) * 0.5) 
        num_test = num_subjects - num_train - num_val 

        np.random.shuffle(all_subjects)

        train_subjects = all_subjects[:num_train]
        val_subjects = all_subjects[num_train:-num_test]
        test_subjects = all_subjects[-num_test:]
        
        # Determine actual subjects used based on dataset type: train, val or test
        if type == "train": 
            self.subjects = train_subjects
        elif type == "val":
            self.subjects = val_subjects
        elif type == "test": 
            self.subjects = test_subjects 

        
    def __len__(self): 
        return len(self.subjects)


    def __getitem__(self, idx):
        result = {}
        image_paths = self.image_paths[self.subject_imageidx_dict[self.subjects[idx]]]
        images = torch.from_numpy(np.array([iio.imread(image_path) / 255. for image_path in image_paths]))
        images = einops.rearrange(images, 'n h w c -> n c h w')
        result["images"] = images

        labels = torch.from_numpy(self.labels[self.subject_imageidx_dict[self.subjects[idx]]]) 
        assert len(np.unique(labels)) == 1 # Labels from the same subject must be the same value!
        assert len(labels) == len(images)
        result["labels"] = torch.from_numpy(np.unique(labels)) # One subject correspond to one label
        return result

class ChildSubjectGroupDatasetPreload(torch.utils.data.Dataset): 
    def __init__(self, root_folder, type="train", view=90): 
        super().__init__()

        np.random.seed(297)

        self.root_path = pathlib.Path(root_folder)
        self.image_root_path = self.root_path / "preload"
        self.label_root_path = self.root_path
        print(f"Using Preload Dataset. Change image root directory from {self.root_path} to " \
              f"{self.image_root_path}")
        print(f"Label root path: {self.label_root_path}")

        image_paths = list(self.root_path.glob("**/*child*rgb*.npy"))
        self.subject_dict = {} 

        # Load all subjects, images and labels 
        
        for image_path in tqdm(image_paths): 
            subject = re.findall(r'child_(\d+)_rgb.npy', image_path.name)[0]

            label_file = list(self.label_root_path.glob(f"**/*{subject}_lbl*.txt"))[0]
            
            lines = open(label_file, "r").readlines()
            height_line = [l for l in lines if "child height" in l]
            assert len(height_line) == 1
            height_line = height_line[0]
            height = float(re.findall(r".*: ([0-9\.]*)", height_line)[0]) 

            self.subject_dict[subject] = [image_path, height]

        all_subjects = sorted(list(self.subject_dict.keys())) 
        # Train-val-test split by 80-10-10 on subjects 
        num_subjects = len(all_subjects)

        num_train = int(num_subjects * 0.8)
        num_val = int((num_subjects - num_train) * 0.5) 
        num_test = num_subjects - num_train - num_val 

        np.random.shuffle(all_subjects)

        train_subjects = all_subjects[:num_train]
        val_subjects = all_subjects[num_train:-num_test]
        test_subjects = all_subjects[-num_test:]
        
        # Determine actual subjects used based on dataset type: train, val or test
        if type == "train": 
            self.subjects = train_subjects
        elif type == "val":
            self.subjects = val_subjects
        elif type == "test": 
            self.subjects = test_subjects 

        
    def __len__(self): 
        return len(self.subjects)


    def __getitem__(self, idx):
        result = {}
        image_file, height = self.subject_dict[self.subjects[idx]]
        with open(image_file, "rb") as f: 
            images = np.load(f)
        result["images"] = torch.from_numpy(images)
        result["labels"] = torch.from_numpy(np.array([height]))
        return result