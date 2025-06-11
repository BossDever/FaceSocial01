"""Face Recognition API Router"""

from fastapi import APIRouter

router = APIRouter()
service = None  # Will be injected by main.py


@router.get("/face-recognition/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "face_recognition"}
