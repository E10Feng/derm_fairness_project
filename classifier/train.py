import time
import numpy as np
from sklearn.model_selection import train_test_split
from dataset import SkinDataset
from torch.utils.data import DataLoader
import torch
import pandas as pd
import copy
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, models
from torchvision.models import VGG16_Weights, ResNet50_Weights
from evaluate_v2 import model_evaluate
from utils import balance_dataset
import matplotlib.pyplot as plt

def model_train(model_name, dataset, learning_rate, save_dir, num_epochs=15, evaluate = False, balance = 'regular', style = 'minority', level = 0.5, experiment_name = any, synth = 'None', device_num = 0):

    ham_image_dir = '/path/to/ham10k'
    ddi_image_dir = '/path/to/ddi'
    fitz_image_dir = '/path/to/fitzpatrick17k/images'
    only_synth_image_dir = '/path/to/only/synthetic/images'
    equal_synth_image_dir = f'/path/to/equally/augmented/synthetic/images_{level}'
    dark_synth_image_dir = f'/path/to/skin/tone/augmented/synthetic/images_{level}'
    dark_positive_synth_image_dir = f'/path/to/disease/augmented/synthetic/images_{level}'

    # Define transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'int_test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'ext_test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
         ])
    }

    if dataset == 'fitz':
        X_train = pd.read_csv('/path/to/train/csv')
        X_test = pd.read_csv('/path/to/test/csv')
        X_val = pd.read_csv('/path/to/val/csv')

        ext_test_metadata = pd.read_csv('/path/to/ddi/csv')
        ext_test_metadata['label'] = ext_test_metadata['label'].apply(lambda x: 1 if x in [True] else 0)

        if balance == 'oversample' and style== 'minority':
            X_train = pd.read_csv(f'/path_to/{level}_disease_oversampled_train.csv')
            fitz_percentage = X_train.groupby('skin_tone')['label'].mean() * 100
            print(f"fitz_os_perc: {fitz_percentage}")
        if balance == 'oversample' and style== 'all':
            X_train = pd.read_csv(f'/path_to/{level}_skintone_oversampled_train.csv')
            fitz_percentage = X_train.groupby('skin_tone')['label'].mean() * 100
            print(f"fitz_os_perc: {fitz_percentage}")

        train_dataset = SkinDataset(X_train, fitz_image_dir, transform=data_transforms['train'])
        val_dataset = SkinDataset(X_val, fitz_image_dir, transform=data_transforms['val'])
        int_test_dataset = SkinDataset(X_test, fitz_image_dir, transform=data_transforms['int_test'])
        ext_test_dataset = SkinDataset(ext_test_metadata, ddi_image_dir, transform=data_transforms['ext_test'])

        if synth == 'only':
            X_train = pd.read_csv('/path/to/only/synth/csv')
            train_dataset = SkinDataset(X_train, only_synth_image_dir, transform=data_transforms['train'])
        if synth == 'equal':  
            X_train = pd.read_csv(f'/path_to/equal_synth_{level}.csv')
            train_dataset = SkinDataset(X_train, equal_synth_image_dir, transform=data_transforms['train'])
        if synth == 'dark':
            X_train = pd.read_csv(f'/path_to/dark_synth_{level}.csv')
            train_dataset = SkinDataset(X_train, dark_synth_image_dir, transform=data_transforms['train'])
        if synth == 'dark_positive':
            X_train = pd.read_csv(f'/path_to/dark_positive_synth_{level}.csv')
            train_dataset = SkinDataset(X_train, dark_positive_synth_image_dir, transform=data_transforms['train'])

    # Create dataloaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    int_test_loader = DataLoader(int_test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    ext_test_loader = DataLoader(ext_test_dataset, batch_size = batch_size, shuffle = False, num_workers = 8)

    dataloaders = {'train': train_loader, 'val': val_loader, 'ext_test': ext_test_loader, 'int_test': int_test_loader}

    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'int_test': len(int_test_dataset), 'ext_test': len(ext_test_dataset)}

    #pick device
    device = torch.device("cuda:"+device_num if torch.cuda.is_available() else "cpu")

    # Load the pre-trained vgg16 model
    if model_name == 'VGG16':
        model_ft = models.vgg16(weights=VGG16_Weights.DEFAULT)
        # Freeze all the first layers
        for param in model_ft.features.parameters():
            param.requires_grad = False

        # Modify the final layer to output two classes
        model_ft.classifier[6] = nn.Sequential(
                        nn.Linear(4096, 256), 
                        nn.ReLU(), 
                        nn.Dropout(0.4),
                        nn.Linear(256, 1),               
        )
    else: 
        model_ft = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256,1)
        )

    
    print(f"{model_name}_{dataset}_{learning_rate}_{num_epochs}_{save_dir}")

    model_ft = model_ft.to(device)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    best_model_wts = copy.deepcopy(model_ft.state_dict())
    best_acc = 0.0
    since = time.time()

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model_ft.train()
            else:
                model_ft.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                optimizer_ft.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model_ft(inputs)
                    preds = torch.sigmoid(outputs).round()
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer_ft.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc.item())

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model_ft.state_dict())

        exp_lr_scheduler.step()
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model_ft.load_state_dict(best_model_wts)

    # Plot training and validation curves 
    plt.figure(figsize=(12, 6)) 
    plt.plot(train_losses, label='Training Loss') 
    plt.plot(val_losses, label='Validation Loss') 
    plt.xlabel('Epochs') 
    plt.ylabel('Loss') 
    plt.legend() 
    plt.title('Training and Validation Loss') 
    plt.show() 
    plt.figure(figsize=(12, 6)) 
    plt.plot(train_accuracies, label='Training Accuracy') 
    plt.plot(val_accuracies, label='Validation Accuracy') 
    plt.xlabel('Epochs') 
    plt.ylabel('Accuracy') 
    plt.legend() 
    plt.title('Training and Validation Accuracy') 
    plt.show()


    if evaluate == True:
        ideal_thr = model_evaluate(model_ft, val_loader, 'val', X_val, dataset, save_dir = save_dir, find_threshold = True, threshold = 0.5, experiment_name=experiment_name, device=device, dataset_sizes=dataset_sizes)
        model_evaluate(model_ft, int_test_loader, 'int_test', X_test, dataset, save_dir=save_dir, find_threshold=False, threshold=ideal_thr,experiment_name=experiment_name, device=device, dataset_sizes=dataset_sizes)
        model_evaluate(model_ft, ext_test_loader, 'ext_test', ext_test_metadata, 'ddi', save_dir =save_dir, find_threshold=False, threshold=ideal_thr,experiment_name=experiment_name, device=device, dataset_sizes=dataset_sizes)

    return model_ft