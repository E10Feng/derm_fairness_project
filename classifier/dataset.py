import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class SkinDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        img_id = self.dataframe.iloc[idx]['image_id']
        img_name = os.path.join(self.image_dir, img_id)

        jpg_path = img_name + '.jpg'
        png_path = img_name

        if os.path.exists(jpg_path):
            img_path = jpg_path 
        elif os.path.exists(png_path):
            img_path = png_path
        else:
            raise FileNotFoundError(f"Image file not found for {img_id}")

        image = Image.open(img_path)
        image = image.convert("RGB")
        label = self.dataframe.iloc[idx]['label']

        if self.transform:
            image = self.transform(image)

        return image, label