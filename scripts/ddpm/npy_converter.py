import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from PIL import Image

# Ground Truth, Original and Destination Folder
gt_folder = "/home/neelesh/crackdiff/datasets/cracktree200/gt"
orig_folder = "/home/neelesh/crackdiff/datasets/cracktree200/orig"
dest_folder = "/home/neelesh/crackdiff/datasets/cracktree200/test"

convert_to_npy = False

files = os.listdir(folder)

# Resizing, converting to npy (if required) and moving to dest folder
for i in tqdm(range(len(files))):
    filename = os.path.join(folder, files[i])

    img = Image.open(filename)
    resized_img = img.resize((256, 256), Image.ANTIALIAS)

    if convert_to_npy:
        resized_img = np.array(resized_img)
        output_filename = os.path.splitext(files[i])[0] + '.npy'
        output_path = os.path.join(dest_folder, output_filename)
        np.save(output_path, resized_img)
    
    else:
        output_filename = files[i]
        output_path = os.path.join(dest_folder, output_filename)
        resized_img.save(output_path)