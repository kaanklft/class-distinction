import webview
import subprocess
import time
import sys

# 1. Arka planda Streamlit uygulamasını başlat
process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "test.py", "--server.headless=true"])

# Streamlit'in ayağa kalkması için 2-3 saniye bekle
time.sleep(3)

# 2. Masaüstü Penceresini Oluştur
window = webview.create_window('Hayvan Sınıflandırma Paneli', 'http://localhost:8501', width=1200, height=800)

# 3. Pencereyi Başlat
webview.start()

# 4. Pencere kapatıldığında arka plandaki Streamlit sürecini de sonlandır
process.kill()