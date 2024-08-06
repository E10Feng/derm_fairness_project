import os

import shutil

import random

import pandas as pd

# This file randomly selects synthetic images to augment all skin tones in the training data
# Returns csv file with augments appended to original csv & a separate image directory 
# Note: you will have to manually copy in the original images into the new directory (simple terminal command)

RATIO = 0.85

# Define the paths
source_dir = '/path/to/synthetic/images'
target_image_dir = f'/path_to_target_dir/equal_synth_images_{RATIO}'
combined_csv_path = f'/path_to_target_csv_dir/equal_synth_{RATIO}.csv'

# Create the target directory if it does not exist
os.makedirs(target_image_dir, exist_ok=True)

# Load the original DataFrame
original_csv_path = '/path/to/original/csv'
original_df = pd.read_csv(original_csv_path)
num_synth_images = int(len(original_df) * RATIO)

# Initialize lists to store data for the CSV file
image_ids = []
disease_labels = []
skin_tone_labels = []

# Function to get disease label and skin tone from folder number
def get_labels(folder_number):
    if 1 <= folder_number <= 6:
        disease_label = '0'
        skin_tone_label = folder_number

    else:
        disease_label = '1'
        skin_tone_label = folder_number - 6

    return disease_label, skin_tone_label

# Loop through each folder and select random images
num_to_add = int(num_synth_images / 12)

for folder_number in range(1, 13):
    folder_path = os.path.join(source_dir, str(folder_number))
    images = os.listdir(folder_path)

    # Select random images from the folder
    selected_images = random.sample(images, num_to_add)

    for image in selected_images:
        source_image_path = os.path.join(folder_path, image)
        target_image_path = os.path.join(target_image_dir, image)

        # Copy image to the new directory
        shutil.copyfile(source_image_path, target_image_path)

        # Get labels
        disease_label, skin_tone_label = get_labels(folder_number)

        # Append data to lists
        image_ids.append(image)
        disease_labels.append(disease_label)
        skin_tone_labels.append(skin_tone_label)

# Create a DataFrame for the new images
new_images_df = pd.DataFrame({
    'image_id': image_ids,
    'label': disease_labels,
    'skin_tone': skin_tone_labels
})

# Combine the original DataFrame with the new images DataFrame
combined_df = pd.concat([original_df, new_images_df])

# Save the combined DataFrame to a new CSV file
combined_df.to_csv(combined_csv_path, index=False)
print(f"Images and combined CSV file have b")