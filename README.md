# 🎭 Face Recognition System

Professional Face Detection, Recognition & Analysis System with GPU optimization and multi-model support.

## ✨ Features

### 🔍 Face Detection
- **Multi-Model Support**: YOLOv9c, YOLOv9e, YOLOv11m
- **Smart Model Selection**: Automatic fallback system
- **GPU Optimization**: CUDA acceleration with memory management
- **Quality Assessment**: Advanced face quality scoring

### 🧠 Face Recognition
- **Multiple Models**: FaceNet, AdaFace, ArcFace
- **High Accuracy**: State-of-the-art embedding extraction
- **Gallery Matching**: Efficient face search and recognition
- **Real-time Processing**: Optimized for production use

### ⚡ Face Analysis
- **Complete Pipeline**: Detection + Recognition in one API
- **Batch Processing**: Multiple images simultaneously  
- **Flexible Configuration**: Customizable analysis modes
- **Performance Monitoring**: Detailed metrics and statistics

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <repository-url>
cd face-recognition-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup System

```bash
# Run setup script
python setup.py
```

This will:
- ✅ Check Python version (3.8+ required)
- ✅ Create necessary directories
- ✅ Install requirements
- ✅ Test imports
- ✅ Check CUDA availability
- ✅ Verify model files

### 3. Start System

```bash
# Quick start
python start.py

# Custom configuration
python start.py --host 0.0.0.0 --port 8080 --workers 1

# Skip system checks (faster startup)
python start.py --skip-checks
```

### 4. Access System

- **Web Interface**: http://localhost:8080
- **API Documentation**: http://localhost:8080/docs
- **Alternative Docs**: http://localhost:8080/redoc
- **Health Check**: http://localhost:8080/health

## 📁 Project Structure

```
face-recognition-system/
├── src/
│   ├── main.py                    # Main application
│   ├── core/
│   │   └── config.py             # Configuration management
│   ├── ai_services/
│   │   ├── common/
│   │   │   └── vram_manager.py   # GPU memory management
│   │   ├── face_detection/       # Face detection models
│   │   ├── face_recognition/     # Face recognition models
│   │   └── face_analysis/        # Integrated analysis
│   └── api/
│       └── complete_endpoints.py # API endpoints
├── model/
│   ├── face-detection/           # YOLO model files
│   └── face-recognition/         # Recognition model files
├── requirements.txt              # Python dependencies
├── setup.py                     # Setup script
├── start.py                     # Startup script
└── README.md                    # This file
```

## 🤖 Model Files

Place the following model files in the correct directories:

### Face Detection Models
```
model/face-detection/
├── yolov9c-face-lindevs.onnx    # ~85MB
├── yolov9e-face-lindevs.onnx    # ~213MB
└── yolov11m-face.pt             # ~40MB
```

### Face Recognition Models
```
model/face-recognition/
├── facenet_vggface2.onnx        # ~94MB
├── adaface_ir101.onnx           # ~261MB
└── arcface_r100.onnx            # ~261MB
```

**Note**: You already have all model files in the correct directories! ✅

## 🔧 API Usage

### Face Detection

```python
import requests
import base64

# Encode image to base64
with open("image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

# Detect faces
response = requests.post("http://localhost:8080/api/face-detection/detect-base64", 
    json={
        "image_base64": image_base64,
        "model_name": "auto",
        "conf_threshold": 0.5,
        "max_faces": 50
    }
)

result = response.json()
print(f"Found {len(result['faces'])} faces")
```

### Face Recognition

```python
# Extract embedding
response = requests.post("http://localhost:8080/api/face-recognition/extract-embedding",
    files={"file": open("face.jpg", "rb")},
    data={"model_name": "facenet"}
)

embedding = response.json()

# Recognize against gallery
gallery = {
    "person_1": {
        "name": "John Doe", 
        "embeddings": [embedding["vector"]]
    }
}

response = requests.post("http://localhost:8080/api/face-recognition/recognize",
    json={
        "face_image_base64": image_base64,
        "gallery": gallery,
        "model_name": "facenet"
    }
)

result = response.json()
```

### Complete Face Analysis

```python
# Upload image for complete analysis
response = requests.post("http://localhost:8080/api/face-analysis/analyze",
    files={"file": open("group_photo.jpg", "rb")},
    data={
        "mode": "full_analysis",
        "gallery_json": json.dumps(gallery),
        "detection_model": "auto",
        "recognition_model": "facenet"
    }
)

analysis = response.json()
print(f"Detected: {analysis['statistics']['total_faces']} faces")
print(f"Recognized: {analysis['statistics']['identified_faces']} people")
```

## ⚙️ Configuration

### Environment Variables

```bash
# API Configuration
export FACE_RECOGNITION_HOST=0.0.0.0
export FACE_RECOGNITION_PORT=8080
export FACE_RECOGNITION_LOG_LEVEL=INFO

# GPU Configuration  
export FACE_RECOGNITION_GPU_ENABLED=true

# Model Directory
export FACE_RECOGNITION_MODEL_DIR=/path/to/models
```

### Configuration Files

Edit `src/core/config.py` for advanced configuration:

```python
# Detection settings
detection_config = {
    "conf_threshold": 0.10,
    "iou_threshold": 0.35,
    "min_quality_threshold": 40,
    # ... more settings
}

# Recognition settings
recognition_config = {
    "similarity_threshold": 0.60,
    "enable_gpu_optimization": True,
    # ... more settings
}
```

## 🔥 GPU Optimization

The system automatically detects and uses CUDA when available:

- **Automatic GPU Detection**: Checks CUDA availability on startup
- **Memory Management**: Intelligent VRAM allocation per model
- **Fallback System**: Graceful CPU fallback when GPU unavailable
- **Multi-GPU Support**: Configurable device selection

### GPU Requirements
- CUDA 11.8 or higher
- 4GB+ VRAM recommended for optimal performance
- Compatible with RTX, GTX, and datacenter GPUs

## 📊 Performance Monitoring

### Health Checks

```bash
# System health
curl http://localhost:8080/health

# Detailed system info
curl http://localhost:8080/system/info

# Model status
curl http://localhost:8080/models/status
```

### Performance Metrics

The system tracks detailed performance metrics:
- Processing times per model
- GPU/CPU usage ratios  
- Memory consumption
- Success/failure rates
- Quality distributions

## 🛠️ Development

### Running in Development Mode

```bash
# With auto-reload
python start.py --reload

# Manual startup
python src/main.py

# Direct uvicorn
uvicorn src.main:app --reload --host 0.0.0.0 --port 8080
```

### Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests (when available)
pytest tests/
```

### Code Quality

```bash
# Type checking
mypy src/

# Code formatting
black src/
isort src/

# Linting
ruff check src/
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Reinstall requirements
   pip install -r requirements.txt --force-reinstall
   ```

2. **CUDA Issues**
   ```bash
   # Check CUDA installation
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Force CPU mode
   export FACE_RECOGNITION_GPU_ENABLED=false
   ```

3. **Model Not Found**
   ```bash
   # Check model files
   python -c "from src.core.config import validate_model_files; print(validate_model_files())"
   ```

4. **Memory Issues**
   ```bash
   # Reduce batch size in config
   # Enable CPU fallback
   # Restart with --workers 1
   ```

### Debug Mode

```bash
# Enable debug logging
export FACE_RECOGNITION_LOG_LEVEL=DEBUG
python start.py
```

## 📈 Performance Tips

1. **GPU Usage**: Enable CUDA for 5-10x speed improvement
2. **Model Selection**: Use YOLOv9c for speed, YOLOv9e for accuracy
3. **Batch Processing**: Process multiple images together
4. **Quality Thresholds**: Adjust thresholds based on use case
5. **Memory Management**: Monitor VRAM usage in production

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`) 
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- YOLOv9 & YOLOv11 for face detection
- FaceNet, AdaFace, ArcFace for face recognition
- FastAPI for the web framework
- PyTorch & ONNX Runtime for model inference

## 📞 Support

- 📧 Email: [your-email@example.com]
- 💬 Issues: GitHub Issues
- 📖 Documentation: [Link to detailed docs]

---

**Made with ❤️ for the computer vision community**

# Face Recognition System Documentation

## Overview
This project is a complete Face Recognition System with detection, recognition, and analysis APIs. It is designed for research, prototyping, and production use, supporting both CPU and GPU (CUDA) environments.

## Features
- Face detection using YOLOv9/YOLOv11 models (ONNX/PT)
- Face recognition using ArcFace, AdaFace, and FaceNet (ONNX)
- Face analysis and quality assessment
- RESTful API endpoints (FastAPI)
- Batch processing and gallery/database management
- CUDA GPU acceleration (if available)
- Modular, extensible, and production-ready codebase

## Quick Start
1. **Clone the repository**
2. **Install Python 3.8+**
3. **Run the setup script:**
   ```bash
   python setup.py
   ```
4. **Download and place model files in the `model/` directory** (see below)
5. **Start the API server:**
   ```bash
   python src/main.py
   # or
   uvicorn src.main:app --host 0.0.0.0 --port 8080 --reload
   ```
6. **Access the API docs:**
   - [http://localhost:8080/docs](http://localhost:8080/docs)

## Directory Structure
```
model/                # Pretrained model files (see below)
output/               # Output results (detections, recognition, analysis)
logs/                 # Log files
src/                  # Source code (FastAPI app, services, routers)
  api/                # API endpoints
  ai_services/        # Core AI logic (detection, recognition, analysis)
  core/               # Config and utilities
```

## Model Files
Place the following files in the `model/` directory:
- `face-detection/yolov9c-face-lindevs.onnx`
- `face-detection/yolov9e-face-lindevs.onnx`
- `face-detection/yolov11m-face.pt`
- `face-recognition/facenet_vggface2.onnx`
- `face-recognition/adaface_ir101.onnx`
- `face-recognition/arcface_r100.onnx`

## API Endpoints
- `/api/face-detection/*` — Face detection
- `/api/face-recognition/*` — Face recognition & gallery management
- `/api/face-analysis/*` — Face analysis & batch processing

See the OpenAPI docs at `/docs` for full details.

## Development & Testing
- All requirements are in `requirements.txt`
- Run tests with `pytest`
- Format code with `black` and `isort`
- Type-check with `mypy`
- Lint with `ruff`

## Troubleshooting
- If CUDA is not available, the system will run in CPU mode (slower)
- If model files are missing, only limited functionality will be available
- For issues, check the logs in the `logs/` directory

## License
MIT License# FaceSocials
# FaceSocial01

🚀 **Face Recognition Social Platform** - ระบบแพลตฟอร์มสื่อสังคมออนไลน์พร้อมเทคโนโลジี AI สำหรับการจดจำใบหน้า

## ✨ คุณสมบัติหลัก

- 🤖 **AI Face Recognition** - ระบบจดจำใบหน้าด้วย AI models (YOLO, FaceNet, ArcFace, AdaFace)
- 🔒 **Face Authentication** - เข้าสู่ระบบด้วยใบหน้า
- 🛡️ **Anti-Spoofing** - ป้องกันการปลอมแปลงใบหน้า
- 👥 **Social Media** - ระบบโซเชียลมีเดียครบวงจร
- 💬 **Real-time Chat** - แชทแบบเรียลไทม์
- 📊 **Age & Gender Analysis** - วิเคราะห์อายุและเพศ
- 🔧 **Admin Dashboard** - ระบบจัดการแอดมิน

## 🏗️ สถาปัตยกรรม

- **Backend**: Python + FastAPI + AI/ML Models
- **Frontend**: TypeScript + React + Next.js  
- **Database**: PostgreSQL + Redis
- **AI Models**: YOLO, FaceNet, AdaFace, ArcFace, DeepFace
- **Infrastructure**: Docker + NVIDIA GPU Support

## 🚀 วิธีการติดตั้งและใช้งาน (One-Click Setup)

### ข้อกำหนดเบื้องต้น
- Docker Desktop (ติดตั้งและเปิดใช้งานแล้ว)
- Git (สำหรับ clone repository)
- อย่างน้อย 8GB RAM
- 10GB พื้นที่ว่างในฮาร์ดดิสก์

### สำหรับ Windows

1. **Clone repository**
   ```cmd
   git clone https://github.com/BossDever/FaceSocial01.git
   cd FaceSocial01
   ```

2. **รันไฟล์ setup (คลิกเดียวเสร็จ)**
   ```cmd
   quick-setup.bat
   ```
   หรือ double-click ที่ไฟล์ `quick-setup.bat`

### สำหรับ Linux/Mac

1. **Clone repository**
   ```bash
   git clone https://github.com/BossDever/FaceSocial01.git
   cd FaceSocial01
   ```

2. **รันไฟล์ setup (คลิกเดียวเสร็จ)**
   ```bash
   chmod +x quick-setup.sh
   ./quick-setup.sh
   ```

## 📋 สิ่งที่ Setup Script จะทำให้อัตโนมัติ

✅ ตรวจสอบ Docker installation  
✅ ตรวจสอบ NVIDIA GPU support  
✅ สร้าง environment configuration  
✅ สร้าง database schema  
✅ ดาวน์โหลด AI models อัตโนมัติ  
✅ Build และ start ทุก services  
✅ รอให้ services พร้อมใช้งาน  
✅ แสดงข้อมูลการเข้าถึงระบบ  

## 🌐 URL การเข้าถึง

หลังจากการติดตั้งเสร็จสิ้น:

- **🖥️ Frontend Application**: http://localhost:3000
- **🔧 Backend API**: http://localhost:8080  
- **📚 API Documentation**: http://localhost:8080/docs
- **🗄️ Database**: localhost:5432
- **🗄️ Redis Cache**: localhost:6379

## 👤 ข้อมูลการเข้าสู่ระบบ

- **👨‍💼 Admin**: username: `admin`, password: `admin123`
- **👤 Demo User**: username: `demo_user`, password: `demo123`

## 🛠️ คำสั่งจัดการระบบ

```bash
# ดูสถานะระบบ
./monitor-system.sh        # Linux/Mac
monitor-system.bat         # Windows

# ดู logs
docker-compose logs -f [service]

# รีสตาร์ทบริการ
docker-compose restart

# หยุดบริการทั้งหมด
docker-compose down

# ลบทุกอย่างเพื่อเริ่มใหม่
docker-compose down -v --rmi all
```

## 📁 โครงสร้างโปรเจกต์

```
FaceSocial01/
├── 🐍 src/                          # Backend Python
│   ├── ai_services/                 # AI Services
│   ├── api/                         # API Endpoints  
│   └── main.py                      # FastAPI App
├── ⚛️ projec-final-fronend/          # Frontend React
│   ├── src/                         # React Components
│   ├── pages/                       # Next.js Pages
│   └── prisma/                      # Database Schema
├── 📚 documentation/                # Documentation
├── 🐳 docker-compose.yml            # Docker Configuration
├── 🚀 quick-setup.sh               # Linux/Mac Setup
├── 🚀 quick-setup.bat              # Windows Setup
└── 📋 README.md                    # This file
```

## 🔧 การพัฒนาและปรับแต่ง

### Development Mode
```bash
# Start in development mode with hot reload
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### Production Deployment
```bash
# Build for production
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up
```

### Manual Model Management
```bash
# Download specific models
python scripts/download_models.py --download

# List available models  
python scripts/download_models.py --list
```

## 🤝 การสนับสนุน

- 📖 **Documentation**: ดูเอกสารฉบับเต็มใน folder `documentation/`
- 🐛 **Issues**: รายงานปัญหาได้ที่ GitHub Issues
- 💡 **Features**: เสนอฟีเจอร์ใหม่ได้ที่ GitHub Discussions

## 📄 License

This project is licensed under the MIT License.

---

**🎯 พร้อมใช้งานแล้ว!** เริ่มต้นด้วยการรัน `quick-setup.bat` (Windows) หรือ `./quick-setup.sh` (Linux/Mac) เพื่อติดตั้งระบบครบวงจรในคลิกเดียว!
