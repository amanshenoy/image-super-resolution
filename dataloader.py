import torch, torch.nn as nn
import torchvision, torchvision.transforms as transforms
import numpy as np 

def load_loader(crop_size: int = 33, batch_size: int = 128, num_workers: int = 1):
    """
    Loads the dataloader of the image directory using the given specifications

    input : crop_size -> image size of the square sub images the model has been trained on
            num_crops -> number of sub-images to consider for each image in the image directory  
    output: dataloader iterable to be able to train on the images
    """
    transform_high_res = transforms.Compose([
            transforms.CenterCrop(48),
            transforms.ToTensor()
        ])
    transform_low_res = transforms.Compose([
            transforms.CenterCrop(48),
            transforms.Resize(36),
            transforms.Resize(48),
            transforms.ToTensor()
        ])
    dataset_high_res = torchvision.datasets.ImageFolder('.', transform = transform_high_res)
    dataset_low_res = torchvision.datasets.ImageFolder('.', transform = transform_low_res)
    dataloader_high_res = torch.utils.data.DataLoader(dataset_high_res, batch_size = batch_size, num_workers = num_workers, shuffle = False)
    dataloader_low_res = torch.utils.data.DataLoader(dataset_low_res, batch_size = batch_size, num_workers = num_workers, shuffle = False)
    return dataloader_low_res, dataloader_high_res