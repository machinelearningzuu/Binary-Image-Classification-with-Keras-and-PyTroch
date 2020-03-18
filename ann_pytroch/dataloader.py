import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils  import shuffle

import torch
from torchvision import datasets, transforms
from PIL import ImageFile
import torch.utils.data

from variables import current_dir,cat_folder,dog_folder, cutoff, workers, batch_size, crop

def load_data():
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    data_transform = transforms.Compose([
            transforms.RandomResizedCrop(crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    data = datasets.ImageFolder(root=current_dir,
                                   transform=data_transform
                                   )

    N = len(data)
    Ntrain = int(len(data) * cutoff)
    print("Size of train dataset is ",Ntrain)
    print("Size of test dataset is ",N - Ntrain)

    shuffled_data = torch.utils.data.Subset(data, np.random.choice(N, N, replace=False))
    train_set = torch.utils.data.Subset(shuffled_data, range(Ntrain))
    test_set  = torch.utils.data.Subset(shuffled_data, range(Ntrain, N))

    train_loader = torch.utils.data.DataLoader(train_set,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=workers
                                                )

    test_loader  = torch.utils.data.DataLoader(test_set,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=workers
                                                )

    return train_loader, test_loader
