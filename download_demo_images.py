# download_demo_images.py
import requests
import os
from PIL import Image
import io

def download_demo_images():
    print("🖼️ Downloading demo chest X-ray images...")
    
    os.makedirs('demo_images', exist_ok=True)
    
    # Public domain chest X-ray images
    demo_urls = {
        'pneumonia_case.jpg': 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Chest_X-ray_-_Pneumonia.jpg/256px-Chest_X-ray_-_Pneumonia.jpg',
        'normal_case.jpg': 'https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Chest_X-ray_-_Normal.jpg/256px-Chest_X-ray_-_Normal.jpg'
    }
    
    for filename, url in demo_urls.items():
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                # Convert to RGB and resize
                img = Image.open(io.BytesIO(response.content))
                img = img.convert('RGB')
                img = img.resize((224, 224))
                img.save(f'demo_images/{filename}')
                print(f"✅ Downloaded {filename}")
            else:
                print(f"❌ Failed to download {filename}")
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
    
    print("🎉 Demo images ready!")

if __name__ == "__main__":
    download_demo_images()
