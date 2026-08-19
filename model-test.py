import time
import librosa
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

# --- 1. CIHAZ VE MODEL HAZIRLIĞI ---
print("Cihaz ve model yükleniyor...")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Çalıştırılan Cihaz: {device.upper()}")

model_id = "OpenMOSS-Team/MOSS-Transcribe-Diarize"

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    trust_remote_code=True,
).to(device)

# --- 2. ISINMA TURU (WARM-UP) ---
# Kronometreyi başlatmadan önce modeli 1 kez sahte/boş bir veriyle çalıştırıyoruz.
print("\nIsınma turu (Warm-up) çalıştırılıyor...")
warmup_audio, _ = librosa.load("ses_10sn.wav", sr=16000)
warmup_prompt = "<|audio_pad|>\nWarmup"
warmup_inputs = processor(
    text=warmup_prompt, audio=warmup_audio, return_tensors="pt"
).to(device)

with torch.no_grad():
    _ = model.generate(**warmup_inputs, max_new_tokens=5)

print("Isınma tamamlandı. Gerçek teste geçiliyor!\n")

# --- 3. ÖLÇÜM LİSTELERİ ---
ses_dosyalari = ["ses_10sn.wav", "ses_30sn.wav"]
sureler = []

# --- 4. KRONOMETRE DÖNGÜSÜ ---
for dosya in ses_dosyalari:
    print(f"{dosya} işleniyor...")

    audio, sr = librosa.load(dosya, sr=16000)

    # Gerçek ölçüm başlangıcı
    baslangic = time.time()

    prompt = "<|audio_pad|>\nTranscribe and diarize the audio."
    inputs = processor(text=prompt, audio=audio, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=500)

    bitis = time.time()
    gecen_sure = bitis - baslangic

    sureler.append(gecen_sure)
    print(f"{dosya} için geçen süre: {gecen_sure:.2f} saniye\n")

# --- 5. GRAFİK ÇİZME (MATPLOTLIB) ---
print("Grafik çizdiriliyor...")

plt.figure(figsize=(8, 5))
plt.bar(ses_dosyalari, sureler, color=["skyblue", "salmon"])

plt.title("MOSS-Transcribe-Diarize: Isınma Sonrası İşlem Süresi")
plt.xlabel("Test Edilen Ses Dosyaları")
plt.ylabel("İşlem Süresi (Saniye)")

for i, sure in enumerate(sureler):
    plt.text(i, sure + 0.1, f"{sure:.2f}s", ha="center")

plt.savefig("sonuc_grafigi_warmup.png")
plt.show()