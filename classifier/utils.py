import numpy as np
import pandas as pd
import os
import matplotlib
from matplotlib import pyplot as plt
from fairlearn.metrics import false_negative_rate, false_positive_rate
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.utils import resample
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0] + flatten(list_of_lists[1:]))
    return list_of_lists[:1] + flatten(list_of_lists[1:])

def find_ideal_threshold(y_true, y_pred):
    """
    Find best threshold to maximize F1 metric 

    PARAMS
    y_true (array): true binary labels
    y_pred (array): predicted probabilities

    RETURNS
    bestthr (float): the best threshold
    """
    p,r,t = precision_recall_curve(y_true, y_pred)
    f1 = np.multiply(2, np.divide(np.multiply(p, r), np.add(r, p)))
    ideal_thr = t[np.where(f1 == max(f1))]

    print(f" max f1 score: {max(f1)}")
    print(f"Best threshold: {ideal_thr}")

    return ideal_thr

def calc_roc_auc(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr,tpr)
    return fpr, tpr, roc_auc

def plot_roc_curve(y_true, y_pred, dataset_name, save_dir):
    fpr, tpr, roc_auc = calc_roc_auc(y_true, y_pred)

    plt.figure()
    plt.plot(fpr,tpr, color = 'darkorange', lw=2, label=f'ROC curve (area = {roc_auc})')
    plt.plot([0,1],[0,1], color = 'navy', lw=2, linestyle='--')
    plt.xlim([0.0,1.0])
    plt.xlim([0.0,1.0])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC for {dataset_name}, area = {roc_auc}')
    plt.legend='lower right'

    save_path = os.path.join(save_dir, f'{dataset_name}_roc_curve.png')
    plt.savefig(save_path)
    plt.show()
    print(f"ROC curve saved as '{save_path}")

def plot_auc_vs_fpr(dataset_name, merged_df, save_dir):

    print(f"unique y_true: {np.unique(merged_df['y_true'])}")
    print(f"unique y_pred: {np.unique(merged_df['y_pred'])}")
    
    overall_fpr = false_positive_rate(merged_df['y_true'], merged_df['y_pred'], pos_label = 1)
    _,_,overall_auc = calc_roc_auc(merged_df['y_true'], merged_df['y_pred'])

    plt.figure()
    plt.xlabel('AUC')
    plt.ylabel('FPR')
    plt.title('AUC vs FPR for each skin tone')
    plt.scatter([overall_auc],[overall_fpr],label=f'{overall_auc:.2f}', color='black', marker = 'x')

    skin_tones = merged_df['skin_tone'].unique()
    skin_tone_fprs = {}

    for skin_tone in skin_tones:
        y_true = merged_df[merged_df['skin_tone'] == skin_tone]['y_true'].values
        y_pred = merged_df[merged_df['skin_tone'] == skin_tone]['y_pred'].values
        fpr = false_positive_rate(y_true, y_pred, pos_label = 1)
        _,_,auc = calc_roc_auc(y_true, y_pred)
        plt.scatter([auc],[fpr],label=f'{skin_tone} (AUC = {auc:.2f})', color='red', marker = 'x')
        plt.text(auc, fpr, skin_tone, fontsize = 9, ha='right')
        skin_tone_fprs[skin_tone] = fpr

    save_path = os.path.join(save_dir, f'{dataset_name}_auc_vs_fpr_curve.png')
    plt.savefig(save_path)
    plt.show()

    max_fpr_st = max(skin_tone_fprs, key=skin_tone_fprs.get)
    min_fpr_st = min(skin_tone_fprs, key=skin_tone_fprs.get)
    max_fpr = skin_tone_fprs[max_fpr_st]
    min_fpr = skin_tone_fprs[min_fpr_st]
    fpr_diff = max_fpr-min_fpr

    save_path2 = os.path.join(save_dir, 'auc_fpr_results.txt')
    with open(save_path2, 'a') as f:
        f.write(f"Results for: {dataset_name}\n")
        f.write(f"Overall AUC: {overall_auc:.4f}\n")
        f.write(f"Overall FPR: {overall_fpr:.4f}\n")
        f.write(f"Skin tone with largest FPR: {max_fpr_st}: {max_fpr:.4f}\n")
        f.write(f"Skin tone with smallest FPR: {min_fpr_st}: {min_fpr:.4f}\n")
        f.write(f"Fairness gap: {fpr_diff:.4f}\n")


    print(f"AUC/FPR txt file saved as {save_path2}")
    print(f"ROC curve saved as '{save_path}")

    return overall_auc, overall_fpr, fpr_diff

def plot_auc_vs_fnr(dataset_name, merged_df, save_dir):

    print(f"unique y_true: {np.unique(merged_df['y_true'])}")
    print(f"unique y_pred: {np.unique(merged_df['y_pred'])}")
    
    overall_fnr = false_negative_rate(merged_df['y_true'], merged_df['y_pred'], pos_label = 1)
    _,_,overall_auc = calc_roc_auc(merged_df['y_true'], merged_df['y_pred'])

    plt.figure()
    plt.xlabel('AUC')
    plt.ylabel('FNR')
    plt.title('AUC vs FNR for each skin tone')
    plt.scatter([overall_auc],[overall_fnr],label=f'{overall_auc:.2f}', color='black', marker = 'x')

    skin_tones = merged_df['skin_tone'].unique()
    skin_tone_fnrs = {}

    for skin_tone in skin_tones:
        y_true = merged_df[merged_df['skin_tone'] == skin_tone]['y_true'].values
        y_pred = merged_df[merged_df['skin_tone'] == skin_tone]['y_pred'].values
        fnr = false_negative_rate(y_true, y_pred, pos_label = 1)
        _,_,auc = calc_roc_auc(y_true, y_pred)
        plt.scatter([auc],[fnr],label=f'{skin_tone} (AUC = {auc:.2f})', color='red', marker = 'x')
        plt.text(auc, fnr, skin_tone, fontsize = 9, ha='right')
        skin_tone_fnrs[skin_tone] = fnr

    save_path = os.path.join(save_dir, f'{dataset_name}_auc_vs_fnr_curve.png')
    plt.savefig(save_path)

    plt.show()

    max_fpr_st = max(skin_tone_fnrs, key=skin_tone_fnrs.get)
    min_fpr_st = min(skin_tone_fnrs, key=skin_tone_fnrs.get)
    max_fpr = skin_tone_fnrs[max_fpr_st]
    min_fpr = skin_tone_fnrs[min_fpr_st]
    fnr_diff = max_fpr-min_fpr

    save_path2 = os.path.join(save_dir, 'auc_fnr_results.txt')
    with open(save_path2, 'a') as f:
        f.write(f"Results for: {dataset_name}\n")
        f.write(f"Overall AUC: {overall_auc:.4f}\n")
        f.write(f"Overall FNR: {overall_fnr:.4f}\n")
        f.write(f"Skin tone with largest FNR: {max_fpr_st}: {max_fpr:.4f}\n")
        f.write(f"Skin tone with smallest FNR: {min_fpr_st}: {min_fpr:.4f}\n")
        f.write(f"Fairness gap: {fnr_diff:.4f}\n")


    print(f"AUC/FNR txt file saved as {save_path2}")

    plt.show()
    print(f"ROC curve saved as '{save_path}")

    return overall_auc, overall_fnr, fnr_diff

def concat(metadata, results_df):

    if metadata.isnull().values.any():
        print("warning: NaN values found in metadata")
    if results_df.isnull().values.any():
        print("warning: NaN values found in results_df")

    metadata.reset_index(drop=True, inplace=True)
    results_df.reset_index(drop=True, inplace=True)

    combined_df = pd.concat([metadata,results_df], axis=1)

    if combined_df.isnull().values.any():
        print("warning: NaN values found after concatenation")

    return combined_df

def undersample(metadata, target_column = 'label'):
    df_majority = metadata[metadata[target_column] == 0]
    df_minority = metadata[metadata[target_column] == 1]

    df_majority_undersampled = df_majority.sample(len(df_minority))

    df_balanced = pd.concat([df_minority, df_majority_undersampled])
    
    return df_balanced

def oversample(metadata, target_column = 'label', ratio = 0.50):
    df_majority = metadata[metadata[target_column] == 0]
    df_minority = metadata[metadata[target_column] == 1]   

    df_minority_oversampled = resample(df_minority, replace = True, n_samples = int(len(df_majority)*ratio), random_state =42) 

    df_balanced = pd.concat([df_majority, df_minority_oversampled])

    return df_balanced

def oversample_data(X, y, sampling_strategy = 'auto', random_state = None):
    ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = ros.fit_resample(X,y)
    resampled_metadata = pd.concat([pd.DataFrame(X_resampled, columns = X.columns), pd.DataFrame(y_resampled, columns = ['label'])], axis =1)
    resampled_counts = resampled_metadata['label'].value_counts()
    print(f'Resampled value counts: {resampled_counts}')
    return resampled_metadata

def undersample_data(X, y, sampling_strategy = 'auto', random_state = None):
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = rus.fit_resample(X,y)
    resampled_metadata = pd.concat([pd.DataFrame(X_resampled, columns = X.columns), pd.DataFrame(y_resampled, columns = ['label'])], axis =1)
    resampled_counts = resampled_metadata['label'].value_counts()
    print(f'Resampled value counts: {resampled_counts}')    
    return resampled_metadata


def balance_dataset(method, X, y):
    if method == 'undersample':
        return undersample_data(X,y)
    elif method == 'oversample':
        return oversample_data(X,y)
    else:
        return X,y
    
def save_metrics_to_csv(experiment_name, save_dir, auc, fpr, fnr, fpr_diff, fnr_diff):
    file_path = os.path.join(save_dir, 'experiment_metrics.csv')

    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write("experiment, AUC, FPR, FNR, FPR gap, FNR gap\n")
    
    with open(file_path, 'a') as f:
        f.write(f"{experiment_name}, {auc}, {fpr}, {fnr}, {fpr_diff}, {fnr_diff}\n")