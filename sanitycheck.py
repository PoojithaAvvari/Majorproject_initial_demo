# src/sanity_check.py
import torch, json
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from dataset import IUDataset
from model import MultimodalClassifier

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    LABELS = json.load(open("data/label_names.json"))
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    ds = IUDataset("data/train_labeled.csv", tokenizer, max_length=128)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)
    batch = next(iter(loader))
    print("Batch keys:", list(batch.keys()))
    print("Image shape:", batch['image'].shape)
    print("Labels shape:", batch['labels'].shape)

    model = MultimodalClassifier(num_labels=len(LABELS)).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(batch['image'].to(device), batch['input_ids'].to(device), batch['attention_mask'].to(device))
    print("Logits shape:", logits.shape)
    print("Sanity check passed.")

if __name__ == "__main__":
    run()
