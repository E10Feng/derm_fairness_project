import os
import shutil
import random
import pandas as pd

# This file creates both an image directory and csv file consisting of only synthetic images

# Define the paths
source_dir = '/path/to/synthetic/images'
target_dir = '/path/to/target/dir'
csv_file = os.path.join(target_dir, 'image_labels.csv')
image_dir = '/path/to/target/image_dir'

# Create the target directory if it does not exist
os.makedirs(target_dir, exist_ok=True)

# images to add 
num_images = 1000

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

# Loop through each folder
for folder_number in range(1, 13):
    folder_path = os.path.join(source_dir, str(folder_number))
    images = os.listdir(folder_path)

    # Select num_images random images from the folder
    selected_images = random.sample(images, num_images)

    for image in selected_images:
        source_image_path = os.path.join(folder_path, image)
        target_image_path = os.path.join(image_dir, image)

        # Copy image to the new directory
        shutil.copyfile(source_image_path, target_image_path)

        # Get labels
        disease_label, skin_tone_label = get_labels(folder_number)

        # Append data to lists
        image_ids.append(image)
        disease_labels.append(disease_label)
        skin_tone_labels.append(skin_tone_label)

# Create a DataFrame and save it as CSV
data = {
    'image_id': image_ids,
    'label': disease_labels,
    'skin_tone': skin_tone_labels
}

df = pd.DataFrame(data)
df.to_csv(csv_file, index=False)
print(f"Images and CSV file have been successfully created in {target_dir}")

