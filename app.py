import webview
import subprocess
import time
import sys

# 1. Start the Streamlit app in the background
process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "test.py", "--server.headless=true"])

# Wait 2-3 seconds for Streamlit to spin up
time.sleep(3)

# 2. Create the desktop window
window = webview.create_window('Animal Classification Panel', 'http://localhost:8501', width=1200, height=800)

# 3. Start the window
webview.start()

# 4. Kill the background Streamlit process when the window is closed
process.kill()
