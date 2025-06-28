"""
Age and Gender Detection Module
"""

from .age_gender_service import AgeGenderDetectionService, get_age_gender_service
from .models import (
    AgeGenderRequest,
    AgeGenderResult,
    PersonAnalysis,
    BatchAgeGenderRequest,
    BatchAgeGenderResult,
    AgeGenderStats,
    ServiceInfo,
    GenderType,
    FaceRegion
)

__all__ = [
    "AgeGenderDetectionService",
    "get_age_gender_service",
    "AgeGenderRequest",
    "AgeGenderResult", 
    "PersonAnalysis",
    "BatchAgeGenderRequest",
    "BatchAgeGenderResult",
    "AgeGenderStats",
    "ServiceInfo",
    "GenderType",
    "FaceRegion"
]
