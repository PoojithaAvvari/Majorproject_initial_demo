# # src/test_infer.py
# import torch, json
# from PIL import Image
# from transformers import AutoTokenizer
# from model import MultimodalClassifier
# from dataset import image_transform
# import numpy as np

# LABELS = json.load(open("data/label_names.json"))
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = MultimodalClassifier(num_labels=len(LABELS)).to(device)
# model.load_state_dict(torch.load("best_model.pt", map_location=device))
# model.eval()

# img = Image.open("C:\\Users\\pooji\\Desktop\\majoprojec t\\archive\\images\\images_normalized\\1_IM-0001-4001.dcm.png").convert("RGB")   # change path
# img_t = image_transform(img).unsqueeze(0).to(device)

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# txt = "The cardiac silhouette and mediastinum size are within normal limits. There is no pulmonary edema. There is no focal consolidation. There are no XXXX of a pleural effusion. There is no evidence of pneumothorax."   # optional
# inputs = tokenizer(txt, padding="max_length", max_length=128,
#                    truncation=True, return_tensors="pt")
# ids  = inputs["input_ids"].to(device)
# mask = inputs["attention_mask"].to(device)

# with torch.no_grad():
#     logits = model(img_t, ids, mask)
#     probs = torch.sigmoid(logits).cpu().numpy()[0]

# for label, p in zip(LABELS, probs):
#     print(label, p)
# # ---- Top-1 prediction regardless of threshold ----
# top1_idx = int(np.argmax(probs))
# top1_label = LABELS[top1_idx]
# top1_prob = float(probs[top1_idx])

# print("\n=== Top-1 prediction (argmax) ===")
# print(f"Label: {top1_label}")
# print(f"Probability: {top1_prob:.4f}")

# # Optional: warn if low confidence
# if top1_prob < 0.2:
#     print("⚠ Model is low-confidence for this sample (max prob < 0.2)")

# # ---- Top-3 predictions (for display) ----
# topk = 3
# topk_idx = np.argsort(probs)[::-1][:topk]

# print(f"\n=== Top-{topk} predictions ===")
# for i in topk_idx:
#     print(f"{LABELS[i]:14s} {probs[i]:.4f}")

# src/test_infer.py
import torch, json
from PIL import Image
from transformers import AutoTokenizer
from model import MultimodalClassifier
from dataset import image_transform
import numpy as np

LABELS = json.load(open("data/label_names.json"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

THRESHOLD = 0.2  # from eval.py best threshold

model = MultimodalClassifier(num_labels=len(LABELS)).to(device)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()

# 1) Load image
img = Image.open("C:\\Users\\pooji\\Downloads\\archive\\images\\images_normalized\\64_IM-2218-4004.dcm.png").convert("RGB")   # change path
img_t = image_transform(img).unsqueeze(0).to(device)

# 2) Text
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
txt = (
    "A XXXX XXXX lung volumes. Lungs are clear without focal airspace disease. No pleural effusions or pneumothoraces. cardiomegaly. Degenerative changes in the spine.,Cardiomegaly with low lung volumes which are grossly clear.,"
    
)
inputs = tokenizer(txt, padding="max_length", max_length=128,
                   truncation=True, return_tensors="pt")
ids  = inputs["input_ids"].to(device)
mask = inputs["attention_mask"].to(device)

# 3) Model inference
with torch.no_grad():
    logits = model(img_t, ids, mask)
    probs = torch.sigmoid(logits).cpu().numpy()[0]

print("=== All probabilities ===")
for label, p in zip(LABELS, probs):
    print(f"{label:14s} {p:.4f}")

# 4) Threshold-based multi-label prediction
positive_labels = [label for label, p in zip(LABELS, probs) if p >= THRESHOLD]

print(f"\n=== Threshold-based predictions (th = {THRESHOLD}) ===")
if positive_labels:
    for lab in positive_labels:
        idx = LABELS.index(lab)
        print(f"{lab:14s} {probs[idx]:.4f}  (PREDICTED)")
else:
    print("No label crosses threshold -> no strong abnormality predicted.")

# 5) Top-1 and Top-3 (for display)
top1_idx = int(np.argmax(probs))
top1_label = LABELS[top1_idx]
top1_prob = float(probs[top1_idx])

print("\n=== Top-1 (argmax) ===")
print(f"{top1_label:14s} {top1_prob:.4f}")
if top1_prob < 0.2:
    print("⚠ Low-confidence prediction (max prob < 0.2).")

topk = 3
topk_idx = np.argsort(probs)[::-1][:topk]
print(f"\n=== Top-{topk} ===")
for i in topk_idx:
    print(f"{LABELS[i]:14s} {probs[i]:.4f}")
