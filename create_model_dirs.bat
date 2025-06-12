@echo off
echo 📁 Creating Model Directory Structure
echo =====================================

REM Create main directories
if not exist "model" mkdir "model"
if not exist "model\face-detection" mkdir "model\face-detection"
if not exist "model\face-recognition" mkdir "model\face-recognition"
if not exist "logs" mkdir "logs"
if not exist "output" mkdir "output"
if not exist "output\detection" mkdir "output\detection"
if not exist "output\recognition" mkdir "output\recognition"
if not exist "output\analysis" mkdir "output\analysis"
if not exist "temp" mkdir "temp"

echo ✅ Directories created!
echo.
echo 📋 Model Files Required:
echo.
echo 🔍 Face Detection Models (place in model\face-detection\):
echo    - yolov9c-face-lindevs.onnx (ประมาณ 85MB)
echo    - yolov9e-face-lindevs.onnx (ประมาณ 213MB)  
echo    - yolov11m-face.pt (ประมาณ 40MB)
echo.
echo 🧠 Face Recognition Models (place in model\face-recognition\):
echo    - facenet_vggface2.onnx (ประมาณ 94MB)
echo    - adaface_ir101.onnx (ประมาณ 261MB)
echo    - arcface_r100.onnx (ประมาณ 261MB)
echo.
echo 💡 หลังจากวาง model files แล้ว ให้รัน start.bat อีกครั้ง
echo.
pause
