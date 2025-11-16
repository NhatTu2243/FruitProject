# app.py — Streamlit Fruit Classifier (with strict "unknown" rejection)
# - Ảnh trái cây: chỉ chấp nhận khi mô hình tự tin cao (precision cao)
# - Vật thể lạ: trả về "không phải trái cây (unknown)"

import json
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# ===================== Cấu hình mặc định =====================
BASE = Path(__file__).resolve().parent
# Mặc định cho Local & Streamlit Community Cloud
DEFAULT_MODEL = Path("/mount/src/fruitproject/outputs_multi/fruit_model.keras") \
    if Path("/mount").exists() else (BASE / "outputs_multi" / "fruit_model.keras")
DEFAULT_CLASSMAP = Path("/mount/src/fruitproject/outputs_multi/class_indices.json") \
    if Path("/mount").exists() else (BASE / "outputs_multi" / "class_indices.json")
DEFAULT_IMG_SIZE = 224

st.set_page_config(page_title="Fruit Classifier", page_icon="🍎", layout="wide")
st.title("🍎🍌🍊 Fruit Classifier — Strict Unknown Rejection")

# ===================== Tiện ích & cache =====================
@st.cache_resource(show_spinner=False)
def load_classes(class_map_path: Path) -> List[str]:
    with open(class_map_path, "r", encoding="utf-8") as f:
        mp = json.load(f)  # {"0":"apple",...}
    # Bảo toàn thứ tự class index 0..C-1
    return [mp[str(i)] for i in range(len(mp))]

@st.cache_resource(show_spinner=True)
def safe_load_model(model_path: Path):
    """
    Load model. Nếu model cũ có Lambda(preprocess_input) thì thêm custom_objects.
    """
    try:
        return tf.keras.models.load_model(model_path)
    except Exception:
        return tf.keras.models.load_model(
            model_path,
            custom_objects={"preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input},
        )

def prepare_image(pil_img: Image.Image, img_size: int = DEFAULT_IMG_SIZE) -> np.ndarray:
    """Resize; KHÔNG chia /255 nếu model đã chuẩn hoá bên trong."""
    img = pil_img.convert("RGB").resize((img_size, img_size))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)  # (1,H,W,3)
    return arr

def predict_raw(model, pil_img: Image.Image, img_size: int) -> np.ndarray:
    x = prepare_image(pil_img, img_size)
    probs = model.predict(x, verbose=0)[0]  # (C,)
    return probs

def topk_from_probs(probs: np.ndarray, classes: List[str], k: int = 3) -> Tuple[str, float, List[Tuple[str,float]]]:
    order = np.argsort(probs)[::-1]
    k = min(k, len(order))
    top_idx = order[:k]
    top_labels = [classes[i] for i in top_idx]
    top_scores = [float(probs[i]) for i in top_idx]
    pred = top_labels[0]
    conf = top_scores[0]
    return pred, conf, list(zip(top_labels, top_scores))

def entropy(probs: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(probs, eps, 1.0)
    return float(-(p * np.log(p)).sum())

def is_unknown(probs: np.ndarray, thr: float, margin_min: float, ent_max: float) -> Tuple[bool, float, float]:
    """Quyết định 'unknown' dựa trên max prob, khoảng cách top1-top2, entropy."""
    order = np.argsort(probs)[::-1]
    top1 = float(probs[order[0]])
    top2 = float(probs[order[1]]) if len(order) > 1 else 0.0
    margin = top1 - top2
    ent = entropy(probs)
    unknown = (top1 < thr) or (margin < margin_min) or (ent > ent_max)
    return unknown, margin, ent

# ===================== Sidebar =====================
st.sidebar.header("⚙️ Cấu hình")
model_path = Path(st.sidebar.text_input("Model file", str(DEFAULT_MODEL)))
classmap_path = Path(st.sidebar.text_input("class_indices.json", str(DEFAULT_CLASSMAP)))
img_size = st.sidebar.number_input("Kích thước ảnh (img_size)", 64, 640, DEFAULT_IMG_SIZE, step=32)
topk = st.sidebar.slider("Top-k hiển thị", 1, 10, 3)

st.sidebar.subheader("🚫 Từ chối dự đoán (unknown)")
strict_mode = st.sidebar.checkbox("Bật chế độ nghiêm ngặt (khuyến nghị)", value=True)
if strict_mode:
    thr = st.sidebar.slider("Ngưỡng xác suất tối thiểu (top-1)", 0.0, 1.0, 0.90, 0.01)
    margin_min = st.sidebar.slider("Ngưỡng chênh lệch top1−top2", 0.0, 1.0, 0.30, 0.01)
    ent_max = st.sidebar.slider("Ngưỡng entropy tối đa", 0.0, 3.0, 1.20, 0.01)
else:
    thr = st.sidebar.slider("Ngưỡng xác suất tối thiểu (top-1)", 0.0, 1.0, 0.70, 0.01)
    margin_min = st.sidebar.slider("Ngưỡng chênh lệch top1−top2", 0.0, 1.0, 0.20, 0.01)
    ent_max = st.sidebar.slider("Ngưỡng entropy tối đa", 0.0, 3.0, 1.60, 0.01)

show_table = st.sidebar.checkbox("Hiện bảng xác suất đầy đủ", value=False)

# Cache: load model & classes
try:
    classes = load_classes(classmap_path)
    model = safe_load_model(model_path)
    st.sidebar.success(f"✅ Đã load: {model_path.name}")
except Exception as e:
    st.sidebar.error(f"❌ Không load được model/class map: {e}")
    st.stop()

st.sidebar.write(f"**Số lớp:** {len(classes)}")
st.sidebar.caption("Lưu ý: chế độ *nghiêm ngặt* ưu tiên chính xác cao (precision), nên có thể bỏ sót một số ảnh trái cây mờ/khó (recall thấp).")

# ===================== Tabs giao diện =====================
tab1, tab2 = st.tabs(["📤 Upload ảnh", "📁 Dự đoán cả thư mục"])

# ---- Tab 1: Upload ảnh ----
with tab1:
    files = st.file_uploader(
        "Chọn 1 hoặc nhiều ảnh (jpg/png/webp/bmp…)",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=True
    )
    if files:
        cols = st.columns(3)
        for i, f in enumerate(files):
            try:
                pil = Image.open(f)
                probs = predict_raw(model, pil, img_size)
                pred, conf, top_list = topk_from_probs(probs, classes, k=topk)
                unk, margin, ent = is_unknown(probs, thr=thr, margin_min=margin_min, ent_max=ent_max)

                with cols[i % 3]:
                    st.image(pil, caption=getattr(f, "name", "uploaded"), use_column_width=True)
                    if unk:
                        st.warning(
                            f"⚠️ **Không phải trái cây (unknown)**  "
                            f"— max conf `{conf:.3f}`, margin `{margin:.3f}`, entropy `{ent:.3f}`"
                        )
                    else:
                        st.success(
                            f"✅ **{pred}** — conf `{conf:.3f}` "
                            f"(margin `{margin:.3f}`, entropy `{ent:.3f}`)"
                        )
                    st.markdown("**Top-k:**")
                    for lbl, sc in top_list:
                        st.write(f"- {lbl}: {sc:.3f}")

                    if show_table:
                        import pandas as pd
                        df_prob = pd.DataFrame(
                            {"class": classes, "probability": probs}
                        ).set_index("class")
                        st.bar_chart(df_prob["probability"])
                        st.caption("Xác suất theo lớp (từ softmax của model).")
            except Exception as e:
                st.warning(f"Lỗi xử lý ảnh: {e}")

# ---- Tab 2: Dự đoán thư mục ----
with tab2:
    st.info("Nhập đường dẫn thư mục ảnh (Windows ví dụ `C:\\Users\\nhatt\\Pictures\\test`) hoặc Linux (`/mount/...`).")
    folder = st.text_input("Đường dẫn thư mục")
    run = st.button("Quét & Dự đoán")
    if run:
        p = Path(folder)
        if not p.exists() or not p.is_dir():
            st.error("Thư mục không tồn tại.")
        else:
            exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            imgs = [fp for fp in p.rglob("*") if fp.suffix.lower() in exts]
            if not imgs:
                st.warning("Không tìm thấy ảnh hợp lệ.")
            else:
                rows = []
                prog = st.progress(0, text="Đang dự đoán…")
                for idx, fp in enumerate(imgs, start=1):
                    try:
                        pil = Image.open(fp)
                        probs = predict_raw(model, pil, img_size)
                        pred, conf, _ = topk_from_probs(probs, classes, k=topk)
                        unk, margin, ent = is_unknown(probs, thr=thr, margin_min=margin_min, ent_max=ent_max)
                        label = "unknown" if unk else pred
                        rows.append((fp.name, str(fp.parent.name), label, conf, margin, ent))
                    except Exception as e:
                        rows.append((fp.name, str(fp.parent.name), f"ERROR: {e}", 0.0, 0.0, 0.0))
                    prog.progress(idx / len(imgs), text=f"{idx}/{len(imgs)} ảnh")

                st.success(f"Đã xử lý {len(rows)} ảnh.")
                import pandas as pd
                df = pd.DataFrame(rows, columns=["filename", "folder", "result", "conf", "margin", "entropy"])
                st.dataframe(df, use_container_width=True)
                st.download_button(
                    "Tải kết quả CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )

st.caption(
    "🔒 Chế độ nghiêm ngặt chỉ chấp nhận khi mô hình rất tự tin. "
    "Nếu muốn nhận diện tốt vật thể lạ hơn nữa, hãy thêm lớp `non_fruit` và huấn luyện lại."
)
