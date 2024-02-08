import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from PIL import Image

# Ground Truth, Original and Destination Folder
gt_folder = "/home/neelesh/crackdiff/datasets/cfd/gt"
orig_folder = "/home/neelesh/crackdiff/datasets/cfd/orig"
dest_folder = "/home/neelesh/crackdiff/datasets/cfd/test"

convert_to_npy = True

if convert_to_npy:
    folder = gt_folder
else:
    folder = orig_folder
files = os.listdir(folder)

# Resizing, converting to npy (if required) and moving to dest folder
for i in tqdm(range(len(files))):
    filename = os.path.join(folder, files[i])
    img = cv2.imread(filename)
    resized_img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_LINEAR)

    if convert_to_npy:
        resized_img[resized_img >= 1] = 255
        output_filename = os.path.splitext(files[i])[0] + '.npy'
        output_path = os.path.join(dest_folder, output_filename)
        np.save(output_path, resized_img)
    
    else:
        output_filename = files[i]
        output_path = os.path.join(dest_folder, output_filename)
        cv2.imwrite(output_path, resized_img)