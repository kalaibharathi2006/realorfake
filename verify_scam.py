import requests
from PIL import Image, ImageDraw
import os

# Create test images
os.makedirs('backend/data/scam_patterns', exist_ok=True)

# 1. Create a "Scam" pattern image
scam_img = Image.new('RGB', (100, 100), color='red')
d = ImageDraw.Draw(scam_img)
d.text((10,10), "SCAM", fill="white")
scam_path = 'backend/data/scam_patterns/test_scam.jpg'
scam_img.save(scam_path)
print(f"Created scam pattern: {scam_path}")

# 2. Create a "Real" (random) image
real_img = Image.new('RGB', (100, 100), color='blue')
d = ImageDraw.Draw(real_img)
d.text((10,10), "REAL", fill="white")
real_path = 'test_real.jpg'
real_img.save(real_path)
print(f"Created real image: {real_path}")

# Default URL
url = 'http://localhost:8000/detect/image'

def test_upload(file_path):
    print(f"\nTesting {file_path}...")
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files)
            if response.status_code == 200:
                print("Result:", response.json())
                return response.json()
            else:
                print("Error:", response.text)
                return None
    except Exception as e:
        print(f"Failed to connect: {e}")
        return None

# Restart backend is needed to load patterns? 
# The ScamMatcher loads on INIT. So we need to restart the server OR 
# checks if the server is running. 
# We assume the user or system handles the server restart. 
# For this verify step, I will assume the server needs to be restarted or 
# I will kill/start it.

# Let's run the test
print("\n--- Sending Requests ---")
test_upload(scam_path) # Should be SCAM
test_upload(real_path) # Should be Real/Fake (not Scam)
