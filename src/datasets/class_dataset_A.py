import os
from this import d
import pandas as pd
from torch.utils import data
from torch.utils.data import Dataset    
from torchvision.io import read_image
import torchvision.transforms as T
import numpy as np
from PIL import Image
import random
import torch

class PlantImageDatasetA(Dataset):
    def __init__(self, 
                csv_file, 
                csv_with_labels, 
                root_dir, 
                main_dir, 
                transform=None, 
                albumentation_transform=None, 
                random_augment=None, 
                imbalance=False):
        
        self.root_dir = root_dir
        self.main_dir = main_dir
        self.csv_file = csv_file
        self.csv_with_labels = csv_with_labels
        self.annotations = pd.read_csv(os.path.join(self.main_dir, self.csv_file))
        self.annotations_with_labels = pd.read_csv(os.path.join(self.main_dir, self.csv_with_labels))
        self.transform = transform
        self.albumentation_transform = albumentation_transform
        self.imbalance = imbalance
        self.random_augment = random_augment

        if self.imbalance:
            self.annotations = self.create_imbalance(self.annotations)

    def create_imbalance(self, dataframe):
        unique_classes = pd.unique(dataframe['Label'])
        k = round(0.5*len(unique_classes))
        classes_to_downsample = np.random.choice(unique_classes, k)
        for class_ in unique_classes:
            if class_ not in classes_to_downsample:
                copied_dataframe = dataframe[dataframe['Label'] == class_]
        imbalanced_dataframe = pd.DataFrame()
        for _class in classes_to_downsample:
            if len(imbalanced_dataframe) == 0:
                imbalanced_dataframe = dataframe[dataframe['Label'] == _class].sample(frac=0.1)
            else:
                imbalanced_dataframe = pd.concat([imbalanced_dataframe, dataframe[dataframe['Label'] == _class]].sample(frac=0.1))
        imbalanced_dataset = pd.concat([imbalanced_dataframe, copied_dataframe])
        return imbalanced_dataset
            
    def __len__(self):
        return (len(self.annotations))
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, f"{self.annotations_with_labels['Label'][index]}/{self.annotations_with_labels['Image'][index]}")
        im = Image.open(img_path)
        image = T.functional.to_tensor(im)
        image_numpy = image.numpy()
        if (self.transform): image = self.transform(image)
        if(self.random_augment): image = self.random_augment(img=image)
        if (self.albumentation_transform): image = self.albumentation_transform(image=image_numpy)
        label = self.annotations['Label'][index]
        return image, label 