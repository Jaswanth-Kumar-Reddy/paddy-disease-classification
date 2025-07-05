# mydatasets.py
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class LeafDiseaseDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(self.annotations['label'].unique())  # We are listing out the sorted list of labels 
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        label_name = self.annotations.iloc[idx]['label']
        filename = self.annotations.iloc[idx]['image_id']
        img_name = os.path.join(self.root_dir, label_name, filename)
        image = Image.open(img_name).convert("RGB")  # Converts the image into RGB
        label = self.class_to_idx[label_name]
        if self.transform:
            image = self.transform(image)
        return image, label
