# 🐾 Class-Ayrim — YOLOv8 ile Görsel Sınıflandırma

YOLOv8 sınıflandırma modeli kullanılarak eğitilen, Streamlit tabanlı bir arayüzle
test edilebilen ve isteğe bağlı olarak masaüstü uygulaması olarak çalıştırılabilen
bir görsel sınıflandırma projesi.

Veri seti [Roboflow Universe](https://universe.roboflow.com/ali-kaan/class-ayrim)
üzerinden alınmıştır.

## 📁 Proje Yapısı

```
class-ayrim/
├── main.py              # Modeli eğitmek için kullanılır (YOLOv8-cls)
├── test.py               # Streamlit web arayüzü (görsel yükle, tahmin al)
├── app.py                # test.py'yi masaüstü penceresinde açan sarmalayıcı (pywebview)
├── requirements.txt       # Gerekli Python paketleri
├── docs/
│   ├── DATASET.md         # Veri seti kaynağı ve lisansı
│   └── ROBOFLOW_EXPORT.md # Roboflow export detayları
└── .gitignore
```

## ⚙️ Kurulum

```bash
git clone https://github.com/<kullanici-adi>/class-ayrim.git
cd class-ayrim
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 📦 Veri Seti

Bu repo veri setini içermez (boyut nedeniyle). Eğitim yapmak için:

1. [Roboflow Universe - class-ayrim](https://universe.roboflow.com/ali-kaan/class-ayrim) sayfasından veri setini indirin (klasör formatında export edin).
2. İndirilen `train/`, `valid/` klasörlerini proje kök dizinine yerleştirin.

Detaylar için `docs/DATASET.md` ve `docs/ROBOFLOW_EXPORT.md` dosyalarına bakabilirsiniz.

## 🏋️ Modeli Eğitme

```bash
python main.py
```

`main.py`, `yolov8n-cls.pt` önceden eğitilmiş ağırlığını başlangıç noktası olarak
kullanır (ilk çalıştırmada Ultralytics tarafından otomatik indirilir) ve eğitim
sonuçlarını `runs/classify/` altına kaydeder.

> **Not:** `device=0` parametresi GPU kullanımı içindir. GPU'nuz yoksa bu satırı
> kaldırın veya `device='cpu'` olarak değiştirin.

## 🖥️ Web Arayüzünde Test Etme

Eğitim tamamlandıktan sonra en iyi ağırlık dosyası `runs/classify/train*/weights/best.pt`
altında oluşur. `test.py` içindeki `model_path` değişkenini kendi klasör adınıza göre
güncelleyin, ardından:

```bash
streamlit run test.py
```

## 🖱️ Masaüstü Uygulaması Olarak Çalıştırma

```bash
python app.py
```

Bu komut, Streamlit sunucusunu arka planda başlatıp `pywebview` ile masaüstü
penceresinde açar.

## 📄 Lisans

Kod: MIT License (isterseniz değiştirebilirsiniz).
Veri seti: CC BY 4.0 — bkz. `docs/DATASET.md`.
