"""Face Recognition API Router"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List, Dict, Any
import logging

router = APIRouter()
service = None  # Will be injected by main.py


@router.get("/face-recognition/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "face_recognition"}


@router.post("/face-recognition/add-face", tags=["Face Recognition"])
async def add_face_endpoint(
    person_name: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Adds a face to the recognition database.
    """
    if not service:
        raise HTTPException(status_code=503, detail="Face Recognition Service not available")

    try:
        image_bytes = await file.read()

        # Assuming service.add_face_from_image expects image_bytes and person_name
        # and returns a dictionary with a 'message' and 'face_id' or similar.
        # This part needs to align with the actual signature and return of your service method.
        result = await service.add_face_from_image(
            image_bytes=image_bytes,
            person_name=person_name,
            person_id=person_name,  # Or generate a unique ID if your service requires it
        )

        if result and result.get("success"):  # Check for a success flag from the service
            return {
                "message": f"Face for {person_name} added successfully.",
                "person_name": person_name,
                "details": result.get("details", "No additional details"),
            }
        else:
            error_message = (
                result.get("error", "Failed to add face due to an unknown error.")
                if result
                else "Failed to add face."
            )
            raise HTTPException(status_code=400, detail=error_message)

    except HTTPException as http_exc:
        # Re-raise HTTPException if it's already one (e.g., from service layer)
        raise http_exc
    except Exception as e:
        logging.getLogger(__name__).error(f"Error in add_face_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Ensure the service is injected in your main.py or wherever you set up the app
# Example (conceptual, adapt to your main.py structure):
# from src.ai_services.face_recognition.face_recognition_service import FaceRecognitionService
# from src.api import face_recognition
#
# recognition_service_instance = FaceRecognitionService(...)
# await recognition_service_instance.initialize()
# face_recognition.service = recognition_service_instance
