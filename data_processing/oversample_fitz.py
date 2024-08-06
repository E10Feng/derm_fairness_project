import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import resample

# This file randomly selects images from training data to be oversampled at chosen ratio
# Two oversampling methods: Skin Tone Oversample (oversample only dark skin tones) & Disease Oversample (oversample only dark skin tones + malignant)
# Returns csv file with augments appended to original csv 

RATIO = 0.35

file_path = '/path/to/training/data'
df = pd.read_csv(file_path)

df_counts = df['label'].value_counts()
print(df_counts)

df_filtered = df[df['skin_tone'].isin([5,6]) & (df['label'] == 1)]
df_filtered_counts = df_filtered['label'].value_counts()
print(df_filtered_counts)

X = df_filtered.drop(columns=['label'])
y = df_filtered['label']

num_samples_to_add = int(len(df_filtered) * RATIO)

X_resampled, y_resampled = resample(X, y, n_samples = num_samples_to_add, random_state=42)

df_resampled = pd.concat([X_resampled, y_resampled],axis=1)

df_resampled_counts = df_resampled['label'].value_counts()
print(df_resampled_counts)

df_combined = pd.concat([df, df_resampled])

df_combined_counts = df_combined['label'].value_counts()
print(df_combined_counts)

combined_file_path = f'/path_to_target/{RATIO}_disease_oversampled_train.csv'
df_combined.to_csv(combined_file_path, index=False)
print('save success')

df_filtered2 = df[df['skin_tone'].isin([5,6])]
X_filtered = df_filtered2.drop(columns=['label'])
y_filtered = df_filtered2['label']

num_samples_to_add2 = int(len(df_filtered2) * RATIO)
X_resampled2, y_resampled2 = resample(X_filtered,y_filtered, n_samples=num_samples_to_add2, random_state=42)
df_resampled2 = pd.concat([X_resampled2,y_resampled2], axis=1)
df_combined2 = pd.concat([df, df_resampled2])

df_combined_counts2 = df_combined2['label'].value_counts()
print(df_combined_counts2)

combined_file_path2 = f'/path_to_target/{RATIO}_skintone_oversampled_train.csv'
df_combined2.to_csv(combined_file_path2, index=False)
print('save success')


