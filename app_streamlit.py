# src/app_streamlit.py
import streamlit as st
import torch, json, os, numpy as np
from transformers import AutoTokenizer
from PIL import Image
from model import MultimodalClassifier
from dataset import image_transform
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

st.set_page_config(layout="wide")
st.title("Multimodal Chest X-Ray Classifier (Image + Text)")

SAMPLE = "/mnt/data/9844485e-d966-405f-b60c-97bd3a653827.png"

LABELS = json.load(open("data/label_names.json"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model = MultimodalClassifier(num_labels=len(LABELS)).to(device)
if os.path.exists("best_model.pt"):
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()

st.sidebar.header("Input Panel")
uploaded = st.sidebar.file_uploader("Upload X-ray image", type=["png","jpg","jpeg"])
use_sample = st.sidebar.checkbox("Use sample image", value=True)
text_input = st.sidebar.text_area("Optional: Add clinical text (Findings/Impression)")

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
elif use_sample and os.path.exists(SAMPLE):
    img = Image.open(SAMPLE).convert("RGB")
else:
    st.stop()

st.image(img, caption="Input Image", use_column_width=True)

# Preprocess
img_t = image_transform(img).unsqueeze(0).to(device)
text = text_input if len(text_input.strip()) > 0 else ""
inputs = tokenizer(text, max_length=128, padding="max_length",
                   truncation=True, return_tensors="pt")
ids  = inputs["input_ids"].to(device)
mask = inputs["attention_mask"].to(device)

# Predict
with torch.no_grad():
    logits = model(img_t, ids, mask)
    probs = torch.sigmoid(logits).cpu().numpy()[0]

st.subheader("Predicted Disease Probabilities")
for label, p in zip(LABELS, probs):
    st.write(f"{label}: **{p:.3f}**")

# GradCAM
st.subheader("GradCAM Heatmap")
top_idx = int(np.argmax(probs))

try:
    target_layer = model.cnn[-1]     # for ResNet18 last conv
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())
    grayscale_cam = cam(input_tensor=img_t, targets=[ClassifierOutputTarget(top_idx)])[0]
    np_img = np.array(img.resize((224,224))) / 255.0
    cam_img = show_cam_on_image(np_img, grayscale_cam, use_rgb=True)
    st.image(cam_img, caption=f"Grad-CAM for: {LABELS[top_idx]}")
except Exception as e:
    st.error(f"GradCAM failed: {e}")
