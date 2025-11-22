# src/test_infer.py
import torch, json
from PIL import Image
from transformers import AutoTokenizer
from model import MultimodalClassifier
from dataset import image_transform

LABELS = json.load(open("data/label_names.json"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultimodalClassifier(num_labels=len(LABELS)).to(device)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()

img = Image.open("C:\\Users\\pooji\\Desktop\\majoprojec t\\archive\\images\\images_normalized\\1_IM-0001-4001.dcm.png").convert("RGB")   # change path
img_t = image_transform(img).unsqueeze(0).to(device)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
txt = "The cardiac silhouette and mediastinum size are within normal limits. There is no pulmonary edema. There is no focal consolidation. There are no XXXX of a pleural effusion. There is no evidence of pneumothorax."   # optional
inputs = tokenizer(txt, padding="max_length", max_length=128,
                   truncation=True, return_tensors="pt")
ids  = inputs["input_ids"].to(device)
mask = inputs["attention_mask"].to(device)

with torch.no_grad():
    logits = model(img_t, ids, mask)
    probs = torch.sigmoid(logits).cpu().numpy()[0]

for label, p in zip(LABELS, probs):
    print(label, p)
