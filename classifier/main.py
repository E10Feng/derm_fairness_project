import time
import torch
import numpy as np
import pandas as pd 
import sys
import os
from train import model_train


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def create_save_dir(base_path = '/base/path', model_name = any, lr = any, epochs = any, balance = any):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        print("Directory was not found so it was created.")
    
    timestamp = time.strftime(f"{model_name}_lr{lr}_{epochs}epochs_{balance}_{style}_level{level}_%Y%m%d-%H%M%S")
    save_dir = os.path.join(base_path, timestamp)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print("Directory was found.")

    return save_dir

if __name__ == '__main__':

    # PARAMS
    dataset = sys.argv[1] # Options: fitz
    mode = sys.argv[2] # Options: train, both
    lr = float(sys.argv[3])
    model_name = sys.argv[4] #Options: VGG16, resnet
    num_epochs = int(sys.argv[5])
    balance = sys.argv[6] #regular, undersample, oversample
    style = sys.argv[7] #minority, all
    level = float(sys.argv[8]) #0.25, 0.50, 0.85, etc.
    synth = sys.argv[9] #none, only, equal, dark, dark_positive
    device_num = sys.argv[10]
    random_seed = int(sys.argv[11])

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    save_dir = create_save_dir(model_name=model_name, lr=lr, epochs=num_epochs, balance=balance)
    experiment_name = f"{balance}_{style}_{level}_{synth}"
    print(experiment_name)

    if mode == 'train':
        model_train(model_name, dataset, lr, save_dir, num_epochs, evaluate=False, balance=balance)
    elif mode == 'both':
        model_train(model_name, dataset, lr, save_dir, num_epochs, evaluate = True, balance=balance, style = style, level = level, experiment_name=experiment_name, synth=synth, device_num = device_num)
        

  

    
        

    