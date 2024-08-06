from __future__ import print_function, division
from matplotlib import pyplot as plt
import torch
import pandas as pd
import numpy as np
import os
import torch.nn as nn
import warnings
from utils import plot_auc_vs_fnr, plot_auc_vs_fpr, find_ideal_threshold, plot_roc_curve, concat
from sklearn.metrics import brier_score_loss

warnings.filterwarnings("ignore")

def model_evaluate(model, dataloader, dataset_size, metadata, dataset_name, save_dir, find_threshold = False, threshold=0.5, experiment_name = any, device = any, dataset_sizes = any):
    model.eval()
    running_loss = 0
    running_corrects = 0
    predictions = []
    true_labels = []
    probabilities = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        with torch.no_grad():
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            
            preds = (probs.cpu().numpy() >= threshold).astype(float)
            #threshold_tensor = torch.tensor(threshold, device=device).expand_as(probs)
            #preds = (probs >= threshold_tensor).float()
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += np.sum(preds == labels.data.cpu().numpy())
        predictions.extend(preds.tolist())
        probabilities.extend(probs.cpu().numpy().tolist())
        true_labels.extend(labels.cpu().numpy().tolist())

    epoch_loss = running_loss / dataset_sizes[dataset_size]
    epoch_acc = running_corrects / dataset_sizes[dataset_size]

    true_labels = np.array(true_labels).flatten()
    predictions = np.array(predictions).flatten()

    brier_score = brier_score_loss(true_labels, predictions)
    print(f'Brier Score: {brier_score}')

    results_df = pd.DataFrame({
        'y_true': true_labels,
        'y_pred': predictions,
        'y-probs': np.array(probabilities).flatten(),
    })
    
    print(f"y_true min: {np.min(true_labels)}, max: {np.max(true_labels)}")
    print(f"y_pred min: {np.min(predictions)}, max: {np.max(predictions)}")
    print(f"y_probs min: {np.min(probabilities)}, max: {np.max(probabilities)}")

    print(f"unique true labels: {np.unique(true_labels)}")
    print(f"unique predictions: {np.unique(predictions)}")

    if not np.all(np.isin(true_labels, [0,1])):
        raise ValueError("y_true has other values besides 0 and 1")
    if not np.all(np.isin(predictions, [0,1])):
        raise ValueError("y_true has other values besides 0 and 1")


    merged_predictions = concat(metadata, results_df)
    save_path = os.path.join(save_dir, f'{dataset_name}_results_df.csv')
    merged_predictions.to_csv(save_path, index = False)
    print(f"merged predictions file saved to {save_path}")
    
    from sklearn.metrics import roc_auc_score
    

    #print(f'{dataset_name} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    print(f'{dataset_name} Loss: {epoch_loss:.4f} AUC: {roc_auc_score(np.array(true_labels), np.array(probabilities)):.4f}')
  
    from utils import save_metrics_to_csv
    #plotting curves
    if not dataset_size == 'val':

        plot_roc_curve(true_labels, probabilities, dataset_name, save_dir)
        if dataset_name == 'fitz':
            all_save_dir = '/path/to/save/dir'

            auc, fpr, fpr_diff = plot_auc_vs_fpr(dataset_name, merged_predictions, save_dir)
            auc, fnr, fnr_diff = plot_auc_vs_fnr(dataset_name, merged_predictions, save_dir)

            save_metrics_to_csv(experiment_name, all_save_dir, auc, fpr, fnr, fpr_diff, fnr_diff)

    if find_threshold:
        #print(np.array(true_labels))
        ideal_thr = find_ideal_threshold(np.array(true_labels), np.array(probabilities))
        return ideal_thr