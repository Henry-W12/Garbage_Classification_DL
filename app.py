# Garbage Classification (ResNet50)
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os

# CONFIG
IMG_SIZE = 224
MODEL_PATH = "outputs/best_model.pth"  # path ke model hasil training
CLASS_NAMES_PATH = "class_names.json"  # akan kita simpan


# MODEL DEFINITION 
@st.cache_resource
def load_model(num_classes):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


# TRANSFORM 
transform = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.1)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# STREAMLIT UI
st.set_page_config(page_title="Garbage Classifier", layout="centered")
st.title("🗑️ Garbage Classification Using ResNet50")
st.write("Upload gambar sampah untuk diklasifikasikan oleh model ResNet50.")

# Load class names
if not os.path.exists(CLASS_NAMES_PATH):
    st.error("class_names.json tidak ditemukan")
    st.stop()

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

model = load_model(len(class_names))

uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, pred = torch.max(probs, 0)

    st.subheader("Prediction")
    st.success(f"Class: {class_names[pred.item()]}")
    st.write(f"Confidence: **{conf.item()*100:.2f}%**")

    st.subheader("All class probabilities")
    for i, cls in enumerate(class_names):
        st.write(f"{cls}: {probs[i].item()*100:.2f}%")

else:
    st.info("Silakan upload gambar untuk memulai")
