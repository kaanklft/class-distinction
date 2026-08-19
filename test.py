import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="Hayvan Sınıflandırma Paneli",
    page_icon="🐾",
    layout="wide"
)

# --- YOLO MODELİNİ YÜKLEME ---
@st.cache_resource
def load_yolo_model():
    # Kendi eğittiğiniz best.pt dosyasının yolunu belirtin
    model_path = 'runs/classify/train-11/weights/best.pt'
    model = YOLO(model_path)
    return model

# Model yükleniyor
try:
    model = load_yolo_model()
except Exception as e:
    st.error(f"Model yüklenirken bir hata oluştu: {e}")
    st.info("Lütfen 'runs/classify/train/weights/best.pt' dosyasının doğru konumda olduğundan emin olun.")

# --- ARAYÜZ (UI) ---
st.title("🐾 Hayvan Sınıflandırma Kontrol Paneli")
st.write("Fotoğraf yükleyin, YOLOv8 modeliniz canlı olarak sınıflandırma yapsın.")

st.divider()

# Fotoğraf Yükleme Alanı
uploaded_file = st.file_uploader(
    "Bir resim seçin veya sürükleyip bırakın...", 
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    # 1. Yüklenen Resmi Aç
    image = Image.open(uploaded_file)

    # 2. Ekranı Sol ve Sağ Olmak Üzere 2 Kolona Böl
    col1, col2 = st.columns([1, 1], gap="large")

    # SOL KOLON: Fotoğrafı Göster
    with col1:
        st.subheader("📷 Yüklenen Fotoğraf")
        st.image(image, use_container_width=True)

    # SAĞ KOLON: Tahmin Sonuçlarını Göster
    with col2:
        st.subheader("YOLOv8 Model Tahmini")
        
        with st.spinner("Görsel analiz ediliyor..."):
            # Ultralytics YOLO modeli PIL imajı doğrudan kabul eder
            results = model(image)
            
            # İlk sonucun verilerini alma
            result = results[0]
            probs = result.probs
            
            # En yüksek olasılıklı sınıf ve güven oranı
            top1_id = probs.top1
            top1_score = probs.top1conf.item()
            predicted_class = result.names[top1_id]
            
            # Tüm sınıfların olasılıkları ve isimleri
            all_scores = probs.data.tolist()  # Tensor'ü listeye çevir
            class_names = result.names        # Sözlük yapısı: {0: 'kedi', 1: 'kopek', ...}

        # Ana Tahmin Kartları
        st.success(f"**Tahmin Edilen Hayvan:** {predicted_class.upper()}")
        st.metric(label="Güven Oranı (Confidence)", value=f"%{top1_score * 100:.2f}")

        st.divider()
        st.write("**Tüm Sınıfların Olasılık Dağılımı:**")

        # Sınıfları yüksek olasılıktan düşüğe doğru sıralayıp listeleme
        sorted_indices = np.argsort(all_scores)[::-1]
        
        for idx in sorted_indices:
            c_name = class_names[idx]
            c_prob = all_scores[idx]
            percentage = c_prob * 100
            
            # Vurgulama: En yüksek tahmini kalın yaz
            if idx == top1_id:
                st.write(f"🎯 **{c_name.capitalize()}**: %{percentage:.2f}")
            else:
                st.write(f"{c_name.capitalize()}: %{percentage:.2f}")
                
            st.progress(float(c_prob))

else:
    st.info("Lütfen sol üstteki alandan test etmek istediğiniz bir resim yükleyin.")