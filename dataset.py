# src/dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import pandas as pd
import ast

IMG_SIZE = 224

image_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class IUDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --------------------------
        # Load image
        # --------------------------
        img_path = row['image_path']
        image = Image.open(img_path).convert('RGB')
        image = image_transform(image)

        # --------------------------
        # Pick impression > findings
        # --------------------------
        text = None
        if 'impression' in row and isinstance(row['impression'], str) and row['impression'].strip():
            text = row['impression']
        elif 'findings' in row and isinstance(row['findings'], str) and row['findings'].strip():
            text = row['findings']
        else:
            text = ""

        # Tokenize text
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # --------------------------
        # Convert labels
        # --------------------------
        labels_vec = ast.literal_eval(row['labels_vec']) if isinstance(row['labels_vec'], str) else row['labels_vec']
        uncertain_vec = ast.literal_eval(row['uncertain_vec']) if isinstance(row['uncertain_vec'], str) else row['uncertain_vec']

        labels = torch.tensor(labels_vec, dtype=torch.float32)
        uncertain = torch.tensor(uncertain_vec, dtype=torch.float32)

        return {
            "image": image,
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels,
            "uncertain": uncertain
        }
