#!/usr/bin/env python3
"""
Model Downloader for AI Services
Downloads required model files from Google Drive if they don't exist locally
"""

import sys
import gdown
from pathlib import Path
from typing import Dict

# Model files configuration with Google Drive URLs
MODEL_CONFIGS = {
    "yolov11n-face.onnx": {
        "url": "https://drive.google.com/file/d/1JHXI0KILRSVTZnErTovuFGBoP-BKfZqS/view?usp=drive_link",
        "folder": "face-detection",
        "size_mb": 6.2
    },
    "yolov11m-face.pt": {
        "url": "https://drive.google.com/file/d/1GQFv9zLGdBkD4JCLeErkBfNylvCOlqAP/view?usp=drive_link",
        "folder": "face-detection",
        "size_mb": 40.8
    },
    "yolov9e-face-lindevs.onnx": {
        "url": "https://drive.google.com/file/d/1z1PbZHoHlcPnuJVKWJbpJ9OBlJ72yo2t/view?usp=drive_link",
        "folder": "face-detection",
        "size_mb": 230.5
    },
    "yolov9c-face-lindevs.onnx": {
        "url": "https://drive.google.com/file/d/1KJfaOv4Kbx0pw-VZWtgV_F1agPaVert9/view?usp=sharing",
        "folder": "face-detection",
        "size_mb": 51.8
    },
    "facenet_vggface2.onnx": {
        "url": "https://drive.google.com/file/d/1E2uMDKxXwQVdZIEWgaLp5Fr9ag6zUvMv/view?usp=sharing",
        "folder": "face-recognition",
        "size_mb": 89.2
    },
    "arcface_r100.onnx": {
        "url": "https://drive.google.com/file/d/1PNbmgXy4bNHOd0xBW35zMG7fjSeIOhBp/view?usp=sharing",
        "folder": "face-recognition",
        "size_mb": 248.6
    },
    "adaface_ir101.onnx": {
        "url": "https://drive.google.com/file/d/1JWIroxhpcIRZ5OktTPgUT7LEAOrE9b_7/view?usp=sharing",
        "folder": "face-recognition",
        "size_mb": 179.3
    }
}

class ModelDownloader:
    def __init__(self, base_model_dir: str = "/app/model"):
        self.base_model_dir = Path(base_model_dir)
        self.ensure_directories()
    
    def ensure_directories(self) -> None:
        """Create necessary model directories"""
        for config in MODEL_CONFIGS.values():
            folder_path = self.base_model_dir / config["folder"]
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Directory created/verified: {folder_path}")
    
    def is_file_exists_and_valid(self, file_path: Path, expected_size_mb: float) -> bool:
        """Check if file exists and has reasonable size"""
        if not file_path.exists():
            return False
        
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        min_size = expected_size_mb * 0.8  # Allow 20% tolerance
        
        if file_size_mb < min_size:
            print(f"⚠️  File {file_path.name} exists but seems incomplete ({file_size_mb:.1f}MB < {min_size:.1f}MB)")
            return False
        
        print(f"✅ File {file_path.name} exists and valid ({file_size_mb:.1f}MB)")
        return True
    
    def extract_file_id_from_url(self, url: str) -> str:
        """Extract Google Drive file ID from URL"""
        if "/file/d/" in url:
            return url.split("/file/d/")[1].split("/")[0]
        elif "id=" in url:
            return url.split("id=")[1].split("&")[0]
        else:
            raise ValueError(f"Cannot extract file ID from URL: {url}")
    
    def download_file(self, filename: str, config: Dict) -> bool:
        """Download a single model file from Google Drive"""
        try:
            file_path = self.base_model_dir / config["folder"] / filename
            
            # Check if file already exists and is valid
            if self.is_file_exists_and_valid(file_path, config["size_mb"]):
                return True
            
            print(f"📥 Downloading {filename} ({config['size_mb']:.1f}MB)...")
            
            # Extract file ID and download using gdown
            file_id = self.extract_file_id_from_url(config["url"])
            download_url = f"https://drive.google.com/uc?id={file_id}"
            
            # Download file
            gdown.download(download_url, str(file_path), quiet=False)
            
            # Verify download
            if self.is_file_exists_and_valid(file_path, config["size_mb"]):
                print(f"✅ Successfully downloaded {filename}")
                return True
            else:
                print(f"❌ Download failed or file incomplete: {filename}")
                return False
                
        except Exception as e:
            print(f"❌ Error downloading {filename}: {str(e)}")
            return False
    
    def download_all_models(self) -> bool:
        """Download all required model files"""
        print("🚀 Starting model download process...")
        print(f"📂 Base model directory: {self.base_model_dir}")
        
        success_count = 0
        total_count = len(MODEL_CONFIGS)
        
        for filename, config in MODEL_CONFIGS.items():
            print(f"\n📋 Processing {filename}...")
            if self.download_file(filename, config):
                success_count += 1
            else:
                print(f"⚠️  Failed to download {filename}")
        
        print(f"\n📊 Download Summary: {success_count}/{total_count} files successful")
        
        if success_count == total_count:
            print("🎉 All models downloaded successfully!")
            return True
        else:
            print("⚠️  Some models failed to download")
            return False
    
    def list_available_models(self) -> None:
        """List all available model files"""
        print("\n📋 Available Model Files:")
        for filename, config in MODEL_CONFIGS.items():
            file_path = self.base_model_dir / config["folder"] / filename
            status = "✅ Available" if file_path.exists() else "❌ Missing"
            print(f"  {filename} ({config['size_mb']:.1f}MB) - {status}")

def main() -> None:
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download AI model files")
    parser.add_argument("--model-dir", default="/app/model", help="Base model directory")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--download", action="store_true", help="Download missing models")
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(args.model_dir)
    
    if args.list:
        downloader.list_available_models()
    
    if args.download:
        success = downloader.download_all_models()
        sys.exit(0 if success else 1)
    
    if not args.list and not args.download:
        # Default behavior: check and download if needed
        downloader.list_available_models()
        print("\n🔍 Checking for missing models...")
        success = downloader.download_all_models()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
