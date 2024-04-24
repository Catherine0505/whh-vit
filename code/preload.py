import os 
import pathlib 
import re
from tqdm import tqdm

import imageio.v3 as iio
import numpy as np 

data_dir = "/home/ml_group/misc/data"
rgb_files = [f for f in os.listdir(data_dir) if ("rgb" in f and "png" in f)] 

subjects = [re.findall(r'child_(\d+)_rgb_\d{3}.png', f)[0] for f in rgb_files]
subjects = np.array(subjects)
subjects = np.unique(subjects)

count = 0
for s in tqdm(subjects): 
    img_files = list(pathlib.Path(data_dir).glob(f"**/child_{s}*rgb*.png")) 
    # print(len(img_files))
    img_arr = [(iio.imread(f) / 255.).transpose(2, 0, 1) for f in img_files]
    img_arr = np.array(img_arr)
    # print(img_arr.shape, img_arr.min(), img_arr.max())

    save_name = "_".join(img_files[0].stem.split("_")[:3]) + ".npy"
    save_path = img_files[0].parent / "preload" / save_name 
    if not os.path.exists(save_path.parent): 
        os.makedirs(save_path.parent, exist_ok=False)
    # print(save_path)
    with open(save_path, "wb") as f: 
        np.save(f, img_arr)