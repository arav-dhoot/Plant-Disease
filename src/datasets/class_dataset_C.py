import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image
import torchvision.transforms as T
import numpy as np
import albumentations as A
import cv2

class PlantImageDatasetC(Dataset):
    def __init__(self, 
                 csv_file, 
                 root_dir, 
                 main_dir, 
                 transform=None, 
                 albumentation_transform=None, 
                 random_augment=None, 
                 imbalance=False):
        
        self.root_dir = root_dir
        self.main_dir = main_dir
        self.csv_file = csv_file
        self.annotations = pd.read_csv(os.path.join(self.main_dir, self.csv_file))
        self.transform = transform
        self.albumentation_transform = albumentation_transform
        self.imbalance = imbalance
        self.random_augment = random_augment

        def create_imbalance(dataframe):
            unique_classes = pd.unique(dataframe['Label'])
            k = round(0.5*len(unique_classes))
            classes_to_downsample = np.random.choice(unique_classes, k)
            for class_ in unique_classes:
                if class_ not in classes_to_downsample:
                    copied_dataframe = dataframe[dataframe['Label'] == class_]
            imbalanced_dataframe = pd.DataFrame()
            for _class in classes_to_downsample:
                if len(imbalanced_dataframe) == 0:
                    imbalanced_dataframe = dataframe[dataframe['Label'] == _class]
                else:
                    imbalanced_dataframe = pd.concat([imbalanced_dataframe, dataframe[dataframe['Label'] == _class]])
            imbalanced_dataframe = imbalanced_dataframe.sample(frac=0.1)
            imbalanced_dataset = pd.concat([imbalanced_dataframe, copied_dataframe])
            return imbalanced_dataset

        if self.imbalance:
            self.annotations = create_imbalance(self.annotations)
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations['Image'][index])
        im = Image.open(img_path)
        image = T.functional.to_tensor(im)
        if(self.transform):
            if(type(self.transform) == T.transforms.Compose):
                image = self.transform(image)
            elif(type(self.transform) == A.core.composition.Compose):
                np_image = np.array(im)
                np_image = np_image.astype(np.uint8)
                image = self.transform(image=np_image)['image']
                image = torch.from_numpy(image)
                image = image.permute(2, 0, 1)  
            image = image.float()
            torch.tensor(image)
        if (self.albumentation_transform):
            # np_image = np.array(im)
            # np_image = np_image.astype(np.uint8)
            # image = self.albumentation_transform(image=np_image)['image']
            # # image = torch.from_numpy(image)
            # image = image.permute(2, 0, 1)
            # image = image.float() 
            # torch.tensor(image)

            np_image = np.array(im)
            np_image = np_image.astype(np.uint8)
            image = A.RandomRain(image=np_image)
            print(image)
            print(image.shape)
            image = torch.from_numpy(image)
            image = image.permute(2, 0, 1)
            image = image.float() 
            image = image.to(torch.float32)
            # image = self.albumentation_transform(image=np.array(image))['image'].permute(2, 0, 1).float()
        if(self.random_augment): image = self.random_augment(image)   
        label = self.annotations['Label'][index]
        return image, label