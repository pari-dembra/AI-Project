import requests
import os
from tqdm import tqdm
import time

def download_file(url, filename, max_retries=3):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, headers=headers, timeout=30)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filename, 'wb') as file, tqdm(
                desc=f"Attempt {attempt + 1}/{max_retries}",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for data in response.iter_content(chunk_size=8192):
                    size = file.write(data)
                    progress_bar.update(size)
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            continue
    return False

if __name__ == "__main__":
    # Multiple mirrors for the model file
    mirrors = [
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        "https://huggingface.co/Ultralytics/yolov8/resolve/main/yolov8n.pt",
        "https://ultralytics.com/assets/yolov8n.pt"
    ]
    
    model_file = "yolov8n.pt"
    print(f"Downloading YOLOv8 model to {model_file}...")
    
    success = False
    for mirror in mirrors:
        print(f"\nTrying mirror: {mirror}")
        if download_file(mirror, model_file):
            success = True
            print("✅ Model downloaded successfully!")
            break
    
    if not success:
        print("\n❌ All download attempts failed.")
        print("\nPlease try these alternative methods:")
        print("1. Visit one of these URLs in your browser:")
        for mirror in mirrors:
            print(f"   - {mirror}")
        print("2. Save the file as 'yolov8n.pt' in the current directory")
        print("\nIf you're having trouble downloading, you can also:")
        print("1. Use a VPN or different network connection")
        print("2. Try downloading from a different device and transfer the file")
        print("3. Check your firewall/antivirus settings") 