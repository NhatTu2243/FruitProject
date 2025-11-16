# app.py — Streamlit Fruit Classifier (with strict "unknown" rejection)
# - Ảnh trái cây: chỉ chấp nhận khi mô hình tự tin cao (precision cao)
# - Vật thể lạ: trả về "không phải trái cây (unknown)"

import json
i# app.py — Demo Streamlit dự đoán trái cây theo ảnh upload
import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Fruit Classifier", page_icon="🍎", layout="centered")

BASE = Path.cwd()
MODEL_PATH = BASE / "outputs_multi" / "fruit_model.keras"   # hoặc .h5 nếu bạn đã convert
CLASSMAP_PATH = BASE / "outputs_multi" / "class_indices.json"
IMG_SIZE = (224,224)

@st.cache_resource
def load_model():
    m = tf.keras.models.load_model(MODEL_PATH)
    return m

@st.cache_resource
def load_classes():
    mp = json.load(open(CLASSMAP_PATH,"r",encoding="utf-8"))
    return [mp[str(i)] for i in range(len(mp))]

model = load_model()
classes = load_classes()

st.title("🍎🍌🍊 Fruit Classifier Demo")
st.caption("Upload ảnh thật để mô hình dự đoán loại trái cây (MobileNetV2 fine-tune).")

files = st.file_uploader("Chọn 1 hoặc nhiều ảnh", type=["jpg","jpeg","png","bmp","webp"], accept_multiple_files=True)

if files:
    for f in files:
        img = Image.open(f).convert("RGB").resize(IMG_SIZE)
        x = np.array(img)[None,...]/255.0
        probs = tf.nn.softmax(model.predict(x, verbose=0), axis=1).numpy()[0]
        idx = int(np.argmax(probs))
        st.image(img, caption=f.name, width=320)
        st.markdown(f"**Dự đoán:** {classes[idx]}  —  **Độ tự tin:** {probs[idx]*100:.2f}%")

        # Top-3
        top3 = probs.argsort()[-3:][::-1]
        st.write("Top-3:")
        for k in top3:
            st.write(f"- {classes[int(k)]}: {probs[int(k)]*100:.2f}%")
        st.divider()


st.caption(
    "🔒 Chế độ nghiêm ngặt chỉ chấp nhận khi mô hình rất tự tin. "
    "Nếu muốn nhận diện tốt vật thể lạ hơn nữa, hãy thêm lớp `non_fruit` và huấn luyện lại."
)
