import os
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset

class PlantImageDatasetC(Dataset):
    def __init__(self, 
                 csv_file, 
                 root_dir, 
                 main_dir, 
                 transform=False, 
                 albumentation_transform=None, 
                 random_augment=None, 
                 imbalance=False):
        
        self.root_dir = root_dir
        self.main_dir = main_dir
        self.csv_file = csv_file
        self.annotations = pd.read_csv(os.path.join(self.main_dir, self.csv_file))
        self.transform = transform
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
        image = Image.open(img_path)
        image_numpy = np.array(image)
        if (self.transform): 
            image = self.transform(T.functional.to_tensor(image))
        if (self.random_augment):
            transform = T.Compose([T.Resize(size = (224, 224)), T.PILToTensor()])
            image = self.random_augment(image)
            image = transform(image)
        label = self.annotations['Label'][index]
        return image, label