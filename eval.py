# src/eval.py
import json, torch, numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset import IUDataset
from model import MultimodalClassifier
from sklearn.metrics import f1_score, roc_auc_score

def main():
    LABELS = json.load(open("data/label_names.json"))
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    ds = IUDataset("data/test_labeled.csv", tokenizer, max_length=128)
    loader = DataLoader(ds, batch_size=8, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalClassifier(num_labels=len(LABELS)).to(device)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.eval()

    all_logits, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device)
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)

            logits = model(imgs, ids, mask).cpu().numpy()
            labels = batch['labels'].numpy()

            all_logits.append(logits)
            all_labels.append(labels)

    all_logits = np.vstack(all_logits)
    all_labels = np.vstack(all_labels)
    probs = 1/(1+np.exp(-all_logits))
    preds = (probs >= 0.5).astype(int)

    print("Macro F1:", f1_score(all_labels, preds, average="macro"))

    try:
        for i, lab in enumerate(LABELS):
            auc = roc_auc_score(all_labels[:, i], probs[:, i])
            print(f"{lab}: AUROC={auc:.4f}")
    except:
        print("AUROC calculation error")

if __name__ == "__main__":
    main()
