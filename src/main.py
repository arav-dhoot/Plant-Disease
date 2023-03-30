import argparse
from datasets import class_dataset_A, class_dataset_B, class_dataset_C
from torch.utils.data import DataLoader
import torchvision.transforms as T
import yaml
import torch
import pytorch_lightning as pl
from model.lightning_module import LitModel
from pytorch_lightning.callbacks.early_stopping import EarlyStopping 
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import time
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm.data import Mixup
from timm.data.auto_augment import rand_augment_transform

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", 
                    help='Either new_plants, plant_disease, or plant_pathology', 
                    choices=['new_plants', 'plant_disease', 'plant_pathology'], 
                    default='new_plants')
parser.add_argument("--epochs", 
                    help='Number of epochs', 
                    default=1, 
                    type=int)
parser.add_argument("--model", 
                    help="Either ResNet or ViT", 
                    choices=['ResNet', 'ViT'], 
                    default='ViT')
parser.add_argument('--learning_rate', 
                    help='Enter a learning rate', 
                    default=0.0003,
                    type=float)
parser.add_argument('--finetune', 
                    action='store_true', 
                    help='Whether to do fine-tuning or not.')
parser.add_argument('--imbalance', 
                    action='store_true', 
                    help='Whether to do fine-tuning or not.')
parser.add_argument('--weight_decay', 
                    help='Enter a weight decay', 
                    default=0.001,
                    type=float)
parser.add_argument('--mixup', 
                    action='store_true',
                    help='Whether to implement mixup or not')

args = parser.parse_args()

BATCH_SIZE = 16
EPOCHS = args.epochs       
NUM_WORKERS = 0
MODEL = args.model
LEARNING_RATE = args.learning_rate  
WEIGHT_DECAY = args.weight_decay
FINE_TUNE = args.finetune
IMBALANCE = args.imbalance
SIZE = (224, 224)
HEIGHT = 224
WIDTH = 224
MIXUP = args.mixup
L = 2
M = 15
MIXUP_ALPHA = 0.5
CUTMIX = 1

valid_transforms = T.Compose([T.Resize(size=SIZE)])
train_transforms = T.Compose([T.Resize(size=SIZE), T.RandomRotation(45),T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomAffine(0, translate=(0.1, 0.1), shear=0.1), T.Normalize(mean=[0.267, 0.267, 0.267], std=[0.247, 0.247, 0.247])])

baseline_transforms = A.Compose([A.Resize(HEIGHT, WIDTH), ToTensorV2()])
advanced_blur_transforms = A.Compose([A.Resize(HEIGHT, WIDTH), A.AdvancedBlur(), ToTensorV2()])
gauss_noise_transforms = A.Compose([A.Resize(HEIGHT, WIDTH), A.GaussNoise(), ToTensorV2()])
down_scale_transforms = A.Compose([A.Resize(HEIGHT, WIDTH), A.Downscale(), ToTensorV2()])
motion_blur_transforms = A.Compose([A.Resize(HEIGHT, WIDTH), A.MotionBlur(), ToTensorV2()])
random_fog_transforms = A.Compose([A.Resize(HEIGHT, WIDTH), A.RandomFog(), ToTensorV2()])
random_rain_transforms = A.Compose([A.Resize(HEIGHT, WIDTH), A.RandomRain(), ToTensorV2()])
random_shadow_transforms = A.Compose([A.Resize(HEIGHT, WIDTH), A.RandomShadow(), ToTensorV2()])
random_sun_flare_transforms = A.Compose([A.Resize(HEIGHT, WIDTH), A.RandomSunFlare(), ToTensorV2()])

with open('../config/config.yaml', 'r') as yaml_file:
    parse_yaml = yaml.safe_load(yaml_file)
    
DATASET = args.dataset
NUM_CLASSES = parse_yaml['num_classes'][DATASET]

if (MIXUP):
        args.mixup = 0.8
        args.cutmix = 1.0
        args.cutmix_minmax = None
        args.mixup_prob = 1.0
        args.mixup_switch_prob = 0.5
        args.mixup_mode = 'batch'
        args.smoothing = 0.1
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=NUM_CLASSES)

if DATASET == "new_plants":

    main_dir = parse_yaml['main_dir'][DATASET]

    train_dataset = class_dataset_A.PlantImageDatasetA(csv_file=parse_yaml['csv'][DATASET]['train'], csv_with_labels=parse_yaml['csv'][DATASET]['train_with_labels'], root_dir=parse_yaml['root_dir'][DATASET]['train'], main_dir=main_dir, transform=train_transforms, imbalance=IMBALANCE)
    valid_dataset = class_dataset_A.PlantImageDatasetA(csv_file=parse_yaml['csv'][DATASET]['valid'], csv_with_labels=parse_yaml['csv'][DATASET]['valid_with_labels'], root_dir=parse_yaml['root_dir'][DATASET]['valid'], main_dir=main_dir, transform=valid_transforms, imbalance=False)
    test_dataset = class_dataset_A.PlantImageDatasetA(csv_file=parse_yaml['csv'][DATASET]['test'], csv_with_labels=parse_yaml['csv'][DATASET]['test_with_labels'], root_dir=parse_yaml['root_dir'][DATASET]['valid'], main_dir=main_dir, transform=valid_transforms, imbalance=False)

    test_dataset_base_line = class_dataset_A.PlantImageDatasetA(csv_file=parse_yaml['csv'][DATASET]['test'], csv_with_labels=parse_yaml['csv'][DATASET]['test_with_labels'], root_dir=parse_yaml['root_dir'][DATASET]['valid'], main_dir=main_dir, albumentation_transform=baseline_transforms, imbalance=False)
    test_dataset_advanced_blur = class_dataset_A.PlantImageDatasetA(csv_file=parse_yaml['csv'][DATASET]['test'], csv_with_labels=parse_yaml['csv'][DATASET]['test_with_labels'], root_dir=parse_yaml['root_dir'][DATASET]['valid'], main_dir=main_dir, albumentation_transform=advanced_blur_transforms, imbalance=False)
    test_dataset_gauss_noise = class_dataset_A.PlantImageDatasetA(csv_file=parse_yaml['csv'][DATASET]['test'], csv_with_labels=parse_yaml['csv'][DATASET]['test_with_labels'], root_dir=parse_yaml['root_dir'][DATASET]['valid'], main_dir=main_dir, albumentation_transform=gauss_noise_transforms, imbalance=False)
    test_dataset_down_scale = class_dataset_A.PlantImageDatasetA(csv_file=parse_yaml['csv'][DATASET]['test'], csv_with_labels=parse_yaml['csv'][DATASET]['test_with_labels'], root_dir=parse_yaml['root_dir'][DATASET]['valid'], main_dir=main_dir, albumentation_transform=down_scale_transforms, imbalance=False)
    test_dataset_motion_blur = class_dataset_A.PlantImageDatasetA(csv_file=parse_yaml['csv'][DATASET]['test'], csv_with_labels=parse_yaml['csv'][DATASET]['test_with_labels'], root_dir=parse_yaml['root_dir'][DATASET]['valid'], main_dir=main_dir, albumentation_transform=motion_blur_transforms, imbalance=False)
    test_dataset_random_fog = class_dataset_A.PlantImageDatasetA(csv_file=parse_yaml['csv'][DATASET]['test'], csv_with_labels=parse_yaml['csv'][DATASET]['test_with_labels'], root_dir=parse_yaml['root_dir'][DATASET]['valid'], main_dir=main_dir, albumentation_transform=random_fog_transforms, imbalance=False)
    test_dataset_random_rain = class_dataset_A.PlantImageDatasetA(csv_file=parse_yaml['csv'][DATASET]['test'], csv_with_labels=parse_yaml['csv'][DATASET]['test_with_labels'], root_dir=parse_yaml['root_dir'][DATASET]['valid'], main_dir=main_dir, albumentation_transform=random_rain_transforms, imbalance=False)
    test_dataset_random_shadow = class_dataset_A.PlantImageDatasetA(csv_file=parse_yaml['csv'][DATASET]['test'], csv_with_labels=parse_yaml['csv'][DATASET]['test_with_labels'], root_dir=parse_yaml['root_dir'][DATASET]['valid'], main_dir=main_dir, albumentation_transform=random_shadow_transforms, imbalance=False)
    test_dataset_random_sun_flare = class_dataset_A.PlantImageDatasetA(csv_file=parse_yaml['csv'][DATASET]['test'], csv_with_labels=parse_yaml['csv'][DATASET]['test_with_labels'], root_dir=parse_yaml['root_dir'][DATASET]['valid'], main_dir=main_dir, albumentation_transform=random_sun_flare_transforms, imbalance=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_loader_base_line = DataLoader(dataset=test_dataset_base_line, batch_size=BATCH_SIZE, shuffle=False) 
    test_loader_advanced_blur = DataLoader(dataset=test_dataset_advanced_blur, batch_size=BATCH_SIZE, shuffle=False) 
    test_loader_gauss_noise = DataLoader(dataset=test_dataset_gauss_noise, batch_size=BATCH_SIZE, shuffle=False)
    test_loader_down_scale = DataLoader(dataset=test_dataset_down_scale, batch_size=BATCH_SIZE, shuffle=False)
    test_loader_motion_blur = DataLoader(dataset=test_dataset_motion_blur, batch_size=BATCH_SIZE, shuffle=False)
    test_loader_random_fog = DataLoader(dataset=test_dataset_random_fog, batch_size=BATCH_SIZE, shuffle=False)
    test_loader_random_rain = DataLoader(dataset=test_dataset_random_rain, batch_size=BATCH_SIZE, shuffle=False)
    test_loader_random_shadow = DataLoader(dataset=test_dataset_random_shadow, batch_size=BATCH_SIZE, shuffle=False)
    test_loader_random_sun_flare = DataLoader(dataset=test_dataset_random_sun_flare, batch_size=BATCH_SIZE, shuffle=False)
    
elif DATASET == 'plant_disease':

    main_dir = parse_yaml['main_dir'][DATASET]

    train_dataset = class_dataset_B.PlantImageDatasetB(csv_file=parse_yaml['csv'][DATASET]['train'], csv_with_labels=parse_yaml['csv'][DATASET]['train_with_labels'], root_dir=parse_yaml['root_dir'][DATASET]['train'], main_dir=main_dir, transform=train_transforms, imbalance=IMBALANCE)
    valid_dataset = class_dataset_B.PlantImageDatasetB(csv_file=parse_yaml['csv'][DATASET]['valid'], csv_with_labels=parse_yaml['csv'][DATASET]['valid_with_labels'], root_dir=parse_yaml['root_dir'][DATASET]['valid'], main_dir=main_dir, transform=valid_transforms, imbalance=False)
    test_dataset = class_dataset_B.PlantImageDatasetB(csv_file=parse_yaml['csv'][DATASET]['test'], csv_with_labels=parse_yaml['csv'][DATASET]['test_with_labels'], root_dir=parse_yaml['root_dir'][DATASET]['test'], main_dir=main_dir, transform=valid_transforms, imbalance=False)

    test_dataset_base_line = class_dataset_B.PlantImageDatasetB(csv_file=parse_yaml['csv'][DATASET]['test'], csv_with_labels=parse_yaml['csv'][DATASET]['test_with_labels'], root_dir=parse_yaml['root_dir'][DATASET]['test'], main_dir=main_dir, albumentation_transform=baseline_transforms, imbalance=False)
    test_dataset_advanced_blur = class_dataset_B.PlantImageDatasetB(csv_file=parse_yaml['csv'][DATASET]['test'], csv_with_labels=parse_yaml['csv'][DATASET]['test_with_labels'], root_dir=parse_yaml['root_dir'][DATASET]['test'], main_dir=main_dir, albumentation_transform=advanced_blur_transforms, imbalance=False)
    test_dataset_gauss_noise = class_dataset_B.PlantImageDatasetB(csv_file=parse_yaml['csv'][DATASET]['test'], csv_with_labels=parse_yaml['csv'][DATASET]['test_with_labels'], root_dir=parse_yaml['root_dir'][DATASET]['test'], main_dir=main_dir, albumentation_transform=gauss_noise_transforms, imbalance=False)
    test_dataset_down_scale = class_dataset_B.PlantImageDatasetB(csv_file=parse_yaml['csv'][DATASET]['test'], csv_with_labels=parse_yaml['csv'][DATASET]['test_with_labels'], root_dir=parse_yaml['root_dir'][DATASET]['test'], main_dir=main_dir, albumentation_transform=down_scale_transforms, imbalance=False)
    test_dataset_motion_blur = class_dataset_B.PlantImageDatasetB(csv_file=parse_yaml['csv'][DATASET]['test'], csv_with_labels=parse_yaml['csv'][DATASET]['test_with_labels'], root_dir=parse_yaml['root_dir'][DATASET]['test'], main_dir=main_dir, albumentation_transform=motion_blur_transforms, imbalance=False)
    test_dataset_random_fog = class_dataset_B.PlantImageDatasetB(csv_file=parse_yaml['csv'][DATASET]['test'], csv_with_labels=parse_yaml['csv'][DATASET]['test_with_labels'], root_dir=parse_yaml['root_dir'][DATASET]['test'], main_dir=main_dir, albumentation_transform=random_fog_transforms, imbalance=False)
    test_dataset_random_rain = class_dataset_B.PlantImageDatasetB(csv_file=parse_yaml['csv'][DATASET]['test'], csv_with_labels=parse_yaml['csv'][DATASET]['test_with_labels'], root_dir=parse_yaml['root_dir'][DATASET]['test'], main_dir=main_dir, albumentation_transform=random_rain_transforms, imbalance=False)
    test_dataset_random_shadow = class_dataset_B.PlantImageDatasetB(csv_file=parse_yaml['csv'][DATASET]['test'], csv_with_labels=parse_yaml['csv'][DATASET]['test_with_labels'], root_dir=parse_yaml['root_dir'][DATASET]['test'], main_dir=main_dir, albumentation_transform=random_shadow_transforms, imbalance=False)
    test_dataset_random_sun_flare = class_dataset_B.PlantImageDatasetB(csv_file=parse_yaml['csv'][DATASET]['test'], csv_with_labels=parse_yaml['csv'][DATASET]['test_with_labels'], root_dir=parse_yaml['root_dir'][DATASET]['test'], main_dir=main_dir, albumentation_transform=random_sun_flare_transforms, imbalance=False)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_loader_base_line = DataLoader(dataset=test_dataset_base_line, batch_size=BATCH_SIZE, shuffle=False)
    test_loader_advanced_blur = DataLoader(dataset=test_dataset_advanced_blur, batch_size=BATCH_SIZE, shuffle=False) 
    test_loader_gauss_noise = DataLoader(dataset=test_dataset_gauss_noise, batch_size=BATCH_SIZE, shuffle=False)
    test_loader_down_scale = DataLoader(dataset=test_dataset_down_scale, batch_size=BATCH_SIZE, shuffle=False)
    test_loader_motion_blur = DataLoader(dataset=test_dataset_motion_blur, batch_size=BATCH_SIZE, shuffle=False)
    test_loader_random_fog = DataLoader(dataset=test_dataset_random_fog, batch_size=BATCH_SIZE, shuffle=False)
    test_loader_random_rain = DataLoader(dataset=test_dataset_random_rain, batch_size=BATCH_SIZE, shuffle=False)
    test_loader_random_shadow = DataLoader(dataset=test_dataset_random_shadow, batch_size=BATCH_SIZE, shuffle=False)
    test_loader_random_sun_flare = DataLoader(dataset=test_dataset_random_sun_flare, batch_size=BATCH_SIZE, shuffle=False)

elif DATASET == 'plant_pathology':

    main_dir = parse_yaml['main_dir'][DATASET]

    train_dataset = class_dataset_C.PlantImageDatasetC(csv_file=parse_yaml['csv'][DATASET]['train'], root_dir=parse_yaml['root_dir'][DATASET], main_dir=main_dir, transform=train_transforms)
    valid_dataset = class_dataset_C.PlantImageDatasetC(csv_file=parse_yaml['csv'][DATASET]['valid'], root_dir=parse_yaml['root_dir'][DATASET], main_dir=main_dir, transform=valid_transforms)
    test_dataset = class_dataset_C.PlantImageDatasetC(csv_file=parse_yaml['csv'][DATASET]['test'], root_dir=parse_yaml['root_dir'][DATASET], main_dir=main_dir, transform=valid_transforms)

    test_dataset_base_line = class_dataset_C.PlantImageDatasetC(csv_file=parse_yaml['csv'][DATASET]['test'], root_dir=parse_yaml['root_dir'][DATASET], main_dir=main_dir, albumentation_transform=baseline_transforms)
    test_dataset_advanced_blur = class_dataset_C.PlantImageDatasetC(csv_file=parse_yaml['csv'][DATASET]['test'], root_dir=parse_yaml['root_dir'][DATASET], main_dir=main_dir, albumentation_transform=advanced_blur_transforms)
    test_dataset_gauss_noise = class_dataset_C.PlantImageDatasetC(csv_file=parse_yaml['csv'][DATASET]['test'], root_dir=parse_yaml['root_dir'][DATASET], main_dir=main_dir, albumentation_transform=gauss_noise_transforms)
    test_dataset_down_scale = class_dataset_C.PlantImageDatasetC(csv_file=parse_yaml['csv'][DATASET]['test'], root_dir=parse_yaml['root_dir'][DATASET], main_dir=main_dir, albumentation_transform=down_scale_transforms)
    test_dataset_motion_blur = class_dataset_C.PlantImageDatasetC(csv_file=parse_yaml['csv'][DATASET]['test'], root_dir=parse_yaml['root_dir'][DATASET], main_dir=main_dir, albumentation_transform=motion_blur_transforms)
    test_dataset_random_fog = class_dataset_C.PlantImageDatasetC(csv_file=parse_yaml['csv'][DATASET]['test'], root_dir=parse_yaml['root_dir'][DATASET], main_dir=main_dir, albumentation_transform=random_fog_transforms)
    test_dataset_random_rain = class_dataset_C.PlantImageDatasetC(csv_file=parse_yaml['csv'][DATASET]['test'], root_dir=parse_yaml['root_dir'][DATASET], main_dir=main_dir, albumentation_transform=random_rain_transforms)
    test_dataset_random_shadow = class_dataset_C.PlantImageDatasetC(csv_file=parse_yaml['csv'][DATASET]['test'], root_dir=parse_yaml['root_dir'][DATASET], main_dir=main_dir, albumentation_transform=random_shadow_transforms)
    test_dataset_random_sun_flare = class_dataset_C.PlantImageDatasetC(csv_file=parse_yaml['csv'][DATASET]['test'], root_dir=parse_yaml['root_dir'][DATASET], main_dir=main_dir, transform=random_sun_flare_transforms)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)   
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False) 

    test_loader_base_line = DataLoader(dataset=test_dataset_base_line, batch_size=BATCH_SIZE, shuffle=False) 
    test_loader_advanced_blur = DataLoader(dataset=test_dataset_advanced_blur, batch_size=BATCH_SIZE, shuffle=False) 
    test_loader_gauss_noise = DataLoader(dataset=test_dataset_gauss_noise, batch_size=BATCH_SIZE, shuffle=False)
    test_loader_down_scale = DataLoader(dataset=test_dataset_down_scale, batch_size=BATCH_SIZE, shuffle=False)
    test_loader_motion_blur = DataLoader(dataset=test_dataset_motion_blur, batch_size=BATCH_SIZE, shuffle=False)
    test_loader_random_fog = DataLoader(dataset=test_dataset_random_fog, batch_size=BATCH_SIZE, shuffle=False)
    test_loader_random_rain = DataLoader(dataset=test_dataset_random_rain, batch_size=BATCH_SIZE, shuffle=False)
    test_loader_random_shadow = DataLoader(dataset=test_dataset_random_shadow, batch_size=BATCH_SIZE, shuffle=False)
    test_loader_random_sun_flare = DataLoader(dataset=test_dataset_random_sun_flare, batch_size=BATCH_SIZE, shuffle=False)

if __name__ == "__main__":
    start = time.time()

    early_stop_callback = EarlyStopping(monitor="valid_loss", min_delta=0.00, patience=5, verbose=False, mode="min")
    
    model_dirpath = os.path.join('saved_models', DATASET, MODEL)
    if FINE_TUNE:
        model_filename = str('with_finetune')
    else:
        model_filename = str('without_finetune')
    
    if(not os.path.exists(model_dirpath)):
            os.makedirs(model_dirpath)
    
    mc = ModelCheckpoint(monitor = 'valid_loss',
                         dirpath = model_dirpath, 
                         filename = model_filename,
                         every_n_epochs=1)

    device = torch.device('cuda')
    model = LitModel(NUM_CLASSES, MODEL, LEARNING_RATE, WEIGHT_DECAY, FINE_TUNE, MIXUP).to(device)
    trainer = pl.Trainer(max_epochs=EPOCHS, accelerator="cuda", callbacks=[early_stop_callback, mc])
    trainer.fit(model, train_loader, valid_loader)

    end = time.time()
    print(f'Time: {end-start}')

    test_results_baseline = trainer.test(dataloaders=test_loader_base_line)
    test_results_advanced_blur = trainer.test(dataloaders=test_loader_advanced_blur)
    test_results_gauss_noise = trainer.test(dataloaders=test_loader_gauss_noise)
    test_results_down_scale = trainer.test(dataloaders=test_loader_down_scale)
    test_results_motion_blur = trainer.test(dataloaders=test_loader_motion_blur)
    test_results_random_fog = trainer.test(dataloaders=test_loader_random_fog)
    test_results_random_rain = trainer.test(dataloaders=test_loader_random_rain)
    test_results_random_shadow = trainer.test(dataloaders=test_loader_random_shadow)
    test_results_random_sun_flare = trainer.test(dataloaders=test_loader_random_sun_flare)
    
    print(f'Baseline results: {test_results_baseline}')
    print(f'Advanced_Blur results:  {test_results_advanced_blur}')
    print(f'Gauss_Noise:  {test_results_gauss_noise}')
    print(f'Down_Scale:  {test_results_down_scale}')
    print(f'Motion_Blur:  {test_results_motion_blur}')
    print(f'Random_Fog:  {test_results_motion_blur}')
    print(f'Random_Rain:  {test_results_random_rain}')
    print(f'Random_Shadow:  {test_results_random_shadow}')
    print(f'Random_Sun_Flare_Up:  {test_results_random_sun_flare}')
