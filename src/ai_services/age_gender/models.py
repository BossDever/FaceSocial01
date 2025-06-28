"""
Data models for Age and Gender Detection Service
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

class GenderType(str, Enum):
    """Gender types"""
    MALE = "Man"
    FEMALE = "Woman"
    UNKNOWN = "Unknown"

class FaceRegion(BaseModel):
    """Face region coordinates"""
    x: int = Field(..., description="X coordinate of face region")
    y: int = Field(..., description="Y coordinate of face region") 
    w: int = Field(..., description="Width of face region")
    h: int = Field(..., description="Height of face region")

class PersonAnalysis(BaseModel):
    """Analysis result for a single person"""
    age: int = Field(..., description="Estimated age")
    gender: str = Field(..., description="Detected gender")
    gender_confidence: float = Field(..., description="Gender confidence score (0-1)")
    face_region: Optional[Dict[str, int]] = Field(None, description="Face bounding box coordinates")

class AgeGenderRequest(BaseModel):
    """Request model for age and gender detection"""
    image: str = Field(..., description="Base64 encoded image data")
    detect_multiple: bool = Field(default=True, description="Detect multiple faces")
    detector_backend: str = Field(default="opencv", description="Face detector backend")

class AgeGenderResult(BaseModel):
    """Result model for age and gender detection"""
    success: bool = Field(..., description="Whether the analysis was successful")
    message: str = Field(..., description="Status message")
    analyses: List[PersonAnalysis] = Field(default=[], description="Analysis results for detected faces")
    total_faces: int = Field(..., description="Total number of faces detected")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")

class BatchAgeGenderRequest(BaseModel):
    """Request model for batch age and gender detection"""
    images: List[str] = Field(..., description="List of base64 encoded images")
    detect_multiple: bool = Field(default=True, description="Detect multiple faces per image")
    detector_backend: str = Field(default="opencv", description="Face detector backend")

class BatchAgeGenderResult(BaseModel):
    """Result model for batch age and gender detection"""
    success: bool = Field(..., description="Whether the batch analysis was successful")
    message: str = Field(..., description="Status message")
    results: List[AgeGenderResult] = Field(default=[], description="Results for each image")
    total_images: int = Field(..., description="Total number of images processed")
    total_faces: int = Field(..., description="Total number of faces detected across all images")
    processing_time: Optional[float] = Field(None, description="Total processing time in seconds")

class AgeGenderStats(BaseModel):
    """Statistics for age and gender analysis"""
    average_age: float = Field(..., description="Average age of detected faces")
    age_range: Dict[str, int] = Field(..., description="Age range (min, max)")
    gender_distribution: Dict[str, int] = Field(..., description="Gender distribution count")
    total_analyzed: int = Field(..., description="Total faces analyzed")

class ServiceInfo(BaseModel):
    """Service information model"""
    service_name: str = Field(..., description="Name of the service")
    version: str = Field(..., description="Service version")
    backend: str = Field(..., description="Backend framework used")
    detector: str = Field(..., description="Face detector backend")
    actions: List[str] = Field(..., description="Available analysis actions")
    initialized: bool = Field(..., description="Whether service is initialized")
    capabilities: List[str] = Field(..., description="Service capabilities")
