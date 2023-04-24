import os
import time
import yaml
import torch
import wandb
import random
import argparse
from timm.data import Mixup
import pytorch_lightning as pl
import torchvision.transforms as T
from torch.utils.data import DataLoader
from model.lightning_module import LitModel
from timm.data.auto_augment import rand_augment_transform
from pytorch_lightning.callbacks.early_stopping import EarlyStopping 
from datasets import class_dataset_A, class_dataset_B, class_dataset_C
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

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
MODEL = args.model
LEARNING_RATE = args.learning_rate  
WEIGHT_DECAY = args.weight_decay
FINE_TUNE = args.finetune
IMBALANCE = args.imbalance
SIZE = (224, 224)
MIXUP = args.mixup

valid_transforms = T.Compose([T.Resize(size=SIZE)])
train_transforms = T.Compose([
    T.Resize(size=SIZE), 
    T.RandomRotation(45),
    T.RandomHorizontalFlip(), 
    T.RandomVerticalFlip(), 
    T.RandomAffine(0, translate=(0.1, 0.1), shear=0.1), 
    T.Normalize(mean=[0.267, 0.267, 0.267], std=[0.247, 0.247, 0.247])]
)

random_augment_transform = rand_augment_transform(
    config_str='rand-m9-mstd0.5', 
    hparams={'translate_const': 117, 'img_mean': (124, 116, 104)}
)

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

    train_dataset = class_dataset_A.PlantImageDatasetA(
                                                       csv_file=parse_yaml['csv'][DATASET]['train'], 
                                                       csv_with_labels=parse_yaml['csv'][DATASET]['train_with_labels'], 
                                                       root_dir=parse_yaml['root_dir'][DATASET]['train'], 
                                                       main_dir=main_dir, 
                                                       transform=train_transforms, 
                                                       imbalance=IMBALANCE
                                                       )
    valid_dataset = class_dataset_A.PlantImageDatasetA(
                                                       csv_file=parse_yaml['csv'][DATASET]['valid'], 
                                                       csv_with_labels=parse_yaml['csv'][DATASET]['valid_with_labels'], 
                                                       root_dir=parse_yaml['root_dir'][DATASET]['valid'], 
                                                       main_dir=main_dir, 
                                                       transform=valid_transforms, 
                                                       imbalance=False
                                                       )
    test_dataset = class_dataset_A.PlantImageDatasetA(
                                                      csv_file=parse_yaml['csv'][DATASET]['test'], 
                                                      csv_with_labels=parse_yaml['csv'][DATASET]['test_with_labels'], 
                                                      root_dir=parse_yaml['root_dir'][DATASET]['valid'], 
                                                      main_dir=main_dir, 
                                                      random_augment=random_augment_transform, 
                                                      imbalance=False
                                                      )

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
elif DATASET == 'plant_disease':

    main_dir = parse_yaml['main_dir'][DATASET]

    train_dataset = class_dataset_B.PlantImageDatasetB(
                                                       csv_file=parse_yaml['csv'][DATASET]['train'], 
                                                       csv_with_labels=parse_yaml['csv'][DATASET]['train_with_labels'], 
                                                       oot_dir=parse_yaml['root_dir'][DATASET]['train'], 
                                                       main_dir=main_dir, 
                                                       transform=train_transforms, 
                                                       imbalance=IMBALANCE
                                                       )
    valid_dataset = class_dataset_B.PlantImageDatasetB(
                                                       csv_file=parse_yaml['csv'][DATASET]['valid'], 
                                                       csv_with_labels=parse_yaml['csv'][DATASET]['valid_with_labels'], 
                                                       root_dir=parse_yaml['root_dir'][DATASET]['valid'], 
                                                       main_dir=main_dir, 
                                                       transform=valid_transforms, 
                                                       imbalance=False
                                                       )
    test_dataset = class_dataset_B.PlantImageDatasetB(
                                                      csv_file=parse_yaml['csv'][DATASET]['test'], 
                                                      csv_with_labels=parse_yaml['csv'][DATASET]['test_with_labels'], 
                                                      root_dir=parse_yaml['root_dir'][DATASET]['test'], 
                                                      main_dir=main_dir, 
                                                      random_augment=random_augment_transform, 
                                                      imbalance=False
                                                      )
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

elif DATASET == 'plant_pathology':

    main_dir = parse_yaml['main_dir'][DATASET]

    train_dataset = class_dataset_C.PlantImageDatasetC(
                                                       csv_file=parse_yaml['csv'][DATASET]['train'], 
                                                       root_dir=parse_yaml['root_dir'][DATASET], 
                                                       main_dir=main_dir, 
                                                       transform=train_transforms, 
                                                       imbalance=IMBALANCE
                                                       )
    valid_dataset = class_dataset_C.PlantImageDatasetC(
                                                       csv_file=parse_yaml['csv'][DATASET]['valid'], 
                                                       root_dir=parse_yaml['root_dir'][DATASET], 
                                                       main_dir=main_dir, 
                                                       transform=valid_transforms, 
                                                       imbalance=False
                                                       )
    test_dataset = class_dataset_C.PlantImageDatasetC(
                                                      csv_file=parse_yaml['csv'][DATASET]['test'], 
                                                      root_dir=parse_yaml['root_dir'][DATASET], 
                                                      main_dir=main_dir, 
                                                      random_augment=random_augment_transform, 
                                                      imbalance=False
                                                      )

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)   
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)    


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
    
    id = random.randint(0, 100)
    config = {
         'epochs': EPOCHS,
         'model': MODEL,
         'learning_rate': LEARNING_RATE,
         'weight_decay': WEIGHT_DECAY, 
    }

    run = wandb.init(
         project='Plant_Disease',
         id=f'{DATASET}-{MODEL}-{WEIGHT_DECAY}',
         config=config,
    )

    device = torch.device('cuda')
    model = LitModel(NUM_CLASSES, MODEL, LEARNING_RATE, WEIGHT_DECAY, FINE_TUNE, MIXUP).to(device)
    trainer = pl.Trainer(max_epochs=EPOCHS, accelerator="cuda", callbacks=[early_stop_callback, mc])
    trainer.fit(model, train_loader, valid_loader)

    end = time.time()
    print(f'Time: {end-start}')

    test_results = trainer.test(dataloaders=test_loader)
    run.log(trainer.callback_metrics())