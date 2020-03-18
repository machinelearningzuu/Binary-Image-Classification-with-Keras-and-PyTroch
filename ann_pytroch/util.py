import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils  import shuffle

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from variables import*

def get_data():
    if not os.path.exists(train_data) or not os.path.exists(test_data):
        print("Data preprocessing and Saving !!!")

        cat_dir = os.path.join(current_dir, cat_folder)
        dog_dir = os.path.join(current_dir, dog_folder)

        cat_image_paths = [os.path.join(cat_dir, image) for image in os.listdir(cat_dir) if os.path.splitext(image)[1]  in ['.jpg','.png']]
        dog_image_paths = [os.path.join(dog_dir, image) for image in os.listdir(dog_dir) if os.path.splitext(image)[1]  in ['.jpg','.png']]

        cat_data = read_image(cat_image_paths,dog_image_paths)
        dog_data = read_image(cat_image_paths,dog_image_paths, cat=False)

        data = np.concatenate((cat_data,dog_data))
        np.random.shuffle(data)

        Ntrain = int((len(cat_data) + len(dog_data)) * cutoff)

        TrainData = data[:Ntrain]
        TestData  = data[Ntrain:]

        np.save(train_data,TrainData)
        np.save(test_data,TestData)

    else:
        print("Data Loading !!!")
        TrainData = np.load(train_data, allow_pickle=True)
        TestData = np.load(test_data, allow_pickle=True)

    return TrainData, TestData

def read_image(cat_image_paths,dog_image_paths,cat=True):
    if cat:
        image_path = cat_image_paths
        anim = "cat"
    else:
        image_path = dog_image_paths
        anim = "dog"
    data = []
    for path in cat_image_paths:
        try:
            img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img=cv2.resize(img, (crop,crop))
            img = img / 255.0
            data.append([np.array(img),encoder[anim]])
        except Exception as e:
            pass
    return np.array(data)
