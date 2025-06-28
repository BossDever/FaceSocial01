#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face Recognition System - Main Application
Fixed version with better import handling
"""

import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List
import asyncio
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Setup logging first
def setup_logging() -> None:
    """Setup logging configuration"""
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/app.log")
        ]
    )
    
    # Silence watchfiles and related loggers
    logging.getLogger("watchfiles").setLevel(logging.WARNING)
    logging.getLogger("watchdog").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Third-party imports
import uvicorn
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.middleware.cors import CORSMiddleware

# Local imports
try:
    from src.core.config import get_settings
    from src.ai_services.common.vram_manager import VRAMManager
    from src.ai_services.face_detection.face_detection_service import (
        FaceDetectionService
    )
    from src.ai_services.face_recognition.face_recognition_service import (
        FaceRecognitionService
    )
    from src.ai_services.face_analysis.face_analysis_service import (
        FaceAnalysisService
    )
    from src.api.complete_endpoints import (
        face_detection_router,
        face_recognition_router,
        face_analysis_router,    )    # Import enhanced API routers (with fallback)
    try:
        from src.api.face_recognition import router as face_recognition_enhanced_router
        from src.api.face_detection import router as face_detection_enhanced_router
        from src.api.face_analysis import router as face_analysis_enhanced_router
        
        # Import anti-spoofing router with individual error handling
        try:
            from src.api.anti_spoofing import router as anti_spoofing_router
            logger.info("✅ Anti-spoofing router imported successfully")
        except ImportError as e:
            logger.warning(f"❌ Anti-spoofing router import failed: {e}")
            anti_spoofing_router = None
        
        # Import age-gender router with individual error handling
        try:
            from src.api.age_gender_endpoints import age_gender_router
            logger.info("✅ Age-gender router imported successfully")
        except ImportError as e:
            logger.warning(f"❌ Age-gender router import failed: {e}")
            age_gender_router = None
            
    except ImportError as router_error:
        logger.warning(f"Core routers import failed: {router_error}")
        # Use fallback routers from complete_endpoints
        face_recognition_enhanced_router = face_recognition_router
        face_detection_enhanced_router = face_detection_router
        face_analysis_enhanced_router = face_analysis_router
        anti_spoofing_router = None
        age_gender_router = None
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    sys.exit(1)

# Custom exception handlers
async def validation_exception_handler_custom(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle validation errors without trying to decode binary data"""
    try:
        # Create a safe version of error details
        safe_errors = []
        for error in exc.errors():
            safe_error = {}
            for key, value in error.items():
                if isinstance(value, bytes):
                    safe_error[key] = f"<bytes: {len(value)} bytes>"
                elif hasattr(value, '__iter__') and not isinstance(value, str):
                    # Handle lists/tuples that might contain bytes
                    safe_items = []
                    for item in value:
                        if isinstance(item, bytes):
                            safe_items.append(f"<bytes: {len(item)} bytes>")
                        else:
                            safe_items.append(str(item))
                    safe_error[key] = ", ".join(safe_items)
                else:
                    safe_error[key] = str(value)
            safe_errors.append(safe_error)
        
        return JSONResponse(
            status_code=422,
            content={"detail": safe_errors}
        )
    except Exception as e:
        logger.error(f"Error in validation exception handler: {e}")
        return JSONResponse(
            status_code=422,
            content={"detail": "Validation error occurred"}
        )

# Helper function to sanitize data for jsonable_encoder
def sanitize_for_jsonable_encoder(data: Any) -> Any:
    """Sanitize data for JSON encoding"""
    if isinstance(data, bytes):
        return data.decode('utf-8', errors='replace')
    elif isinstance(data, str):
        return data.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
    elif isinstance(data, list):
        return [sanitize_for_jsonable_encoder(item) for item in data]
    elif isinstance(data, dict):
        return {
            sanitize_for_jsonable_encoder(k): sanitize_for_jsonable_encoder(v)
            for k, v in data.items()
        }
    return data

async def fastapi_native_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Custom exception handler for request validation errors."""
    
    original_errors = exc.errors()
    original_body = exc.body

    sanitized_errors = sanitize_for_jsonable_encoder(original_errors)
    sanitized_body = sanitize_for_jsonable_encoder(original_body)

    logger.warning(
        f"Validation error for request {request.url}. "
        f"Sanitized errors: {sanitized_errors}. "
        f"Body type after sanitization: {type(sanitized_body)}"
    )

    return JSONResponse(
        status_code=422,
        content=jsonable_encoder({
            "detail": sanitized_errors,
            "body": sanitized_body
        })
    )

# Service names constants
VRAM_MANAGER_SERVICE = "vram_manager"
FACE_DETECTION_SERVICE = "face_detection_service"
FACE_RECOGNITION_SERVICE = "face_recognition_service"
FACE_ANALYSIS_SERVICE = "face_analysis_service"

ALL_SERVICE_KEYS: List[str] = [
    VRAM_MANAGER_SERVICE,
    FACE_DETECTION_SERVICE,
    FACE_RECOGNITION_SERVICE,
    FACE_ANALYSIS_SERVICE,
]

USER_FACING_SERVICE_KEYS: List[str] = [
    FACE_DETECTION_SERVICE,
    FACE_RECOGNITION_SERVICE,
    FACE_ANALYSIS_SERVICE,
]

async def initialize_single_service(
    app: FastAPI,
    service_name: str,
    service_class: Any,
    vram_manager: VRAMManager,
    config: Any
) -> bool:
    """Initialize a single service"""
    logger.info(f"🔧 Initializing {service_class.__name__}...")
    service_instance = service_class(vram_manager=vram_manager, config=config)
    if await service_instance.initialize():
        setattr(app.state, service_name, service_instance)
        logger.info(f"✅ {service_class.__name__} initialized")
        return True
    logger.error(f"❌ {service_class.__name__} initialization failed")
    return False

async def initialize_services(app: FastAPI, settings: Any) -> bool:
    """Initialize all services and attach to app.state"""
    try:
        # Create required directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("output/detection", exist_ok=True)
        os.makedirs("output/recognition", exist_ok=True)
        os.makedirs("output/analysis", exist_ok=True)
        os.makedirs("temp", exist_ok=True)

        # Initialize VRAM Manager
        logger.info("🔧 Initializing VRAM Manager...")
        vram_manager = VRAMManager(settings.vram_config)
        setattr(app.state, VRAM_MANAGER_SERVICE, vram_manager)
        logger.info("✅ VRAM Manager initialized")

        # Services configuration
        services_to_init = {
            FACE_DETECTION_SERVICE: (
                FaceDetectionService,
                settings.detection_config
            ),
            FACE_RECOGNITION_SERVICE: (
                FaceRecognitionService,
                settings.recognition_config
            ),
            FACE_ANALYSIS_SERVICE: (
                FaceAnalysisService,
                settings.analysis_config
            ),
        }

        # Initialize services
        init_tasks = []
        for service_name, (service_class, service_config) in services_to_init.items():
            init_tasks.append(
                initialize_single_service(
                    app, service_name, service_class, vram_manager, service_config
                )
            )
        
        results = await asyncio.gather(*init_tasks)
        if not all(results):
            logger.error("❌ One or more services failed to initialize.")
            return False

        # Set shared services for Face Analysis Service
        face_analysis_service = getattr(app.state, FACE_ANALYSIS_SERVICE, None)
        face_detection_service = getattr(app.state, FACE_DETECTION_SERVICE, None) 
        face_recognition_service = getattr(app.state, FACE_RECOGNITION_SERVICE, None)
        
        if face_analysis_service and face_detection_service and face_recognition_service:
            logger.info("🔧 Setting shared services for Face Analysis Service...")
            face_analysis_service.set_shared_services(
                face_detection_service=face_detection_service,
                face_recognition_service=face_recognition_service
            )
            logger.info("✅ Shared services set for Face Analysis Service")

        logger.info("✅ All services initialized successfully.")
        return True

    except Exception as e:
        logger.exception(f"❌ Critical error during service initialization: {e}")
        return False

async def shutdown_services(app: FastAPI) -> None:
    """Gracefully shutdown all services"""
    logger.info("🔌 Shutting down services...")
    
    shutdown_tasks = []
    for service_key in ALL_SERVICE_KEYS:
        service = getattr(app.state, service_key, None)
        if service and hasattr(service, 'shutdown') and callable(service.shutdown):
            logger.info(f"Shutting down {service_key}...")
            shutdown_tasks.append(service.shutdown())
        elif service:
            logger.info(f"{service_key} does not have a callable shutdown method.")

    if shutdown_tasks:
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        logger.info("✅ Services shut down.")
    else:
        logger.info("No services required explicit shutdown.")

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan events (startup and shutdown)."""
    settings = get_settings()
    app.state.settings = settings

    if not await initialize_services(app, settings):
        logger.critical(
            "Service initialization failed. Application will not start properly."
        )
    
    yield  # Application is now running

    await shutdown_services(app)

# Create FastAPI app instance with lifespan management
app = FastAPI(
    title="Face Recognition API",
    description="API for face detection, recognition, and analysis.",
    version="1.1.0",
    lifespan=lifespan,
    exception_handlers={
        RequestValidationError: fastapi_native_validation_exception_handler
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(face_detection_router, prefix="/api", tags=["Face Detection"])
app.include_router(face_recognition_router, prefix="/api", tags=["Face Recognition"])
app.include_router(face_analysis_router, prefix="/api", tags=["Face Analysis"])

# Include enhanced API routers
app.include_router(face_recognition_enhanced_router, prefix="/api/face-recognition", tags=["Face Recognition Enhanced"])
app.include_router(face_detection_enhanced_router, prefix="/api/face-detection", tags=["Face Detection Enhanced"])
app.include_router(face_analysis_enhanced_router, prefix="/api/face-analysis", tags=["Face Analysis Enhanced"])

# Include Fast Face Detection API (Simple YOLOv11m)
try:
    from .api_routes.fast_face_detection import router as fast_face_detection_router
    app.include_router(fast_face_detection_router, prefix="/api/face-detection", tags=["Fast Face Detection"])
    logger.info("✅ Fast Face Detection API endpoints registered")
except ImportError as e:
    logger.warning(f"⚠️ Fast Face Detection API not available: {e}")
except Exception as e:
    logger.error(f"❌ Failed to register Fast Face Detection API: {e}")

# Include Optimized Face Detection API (Advanced optimizations)
try:
    from .api_routes.optimized_face_detection import router as optimized_face_detection_router
    app.include_router(optimized_face_detection_router, tags=["Optimized Face Detection"])
    logger.info("✅ Optimized Face Detection API endpoints registered")
except ImportError as e:
    logger.warning(f"⚠️ Optimized Face Detection API not available: {e}")
except Exception as e:
    logger.error(f"❌ Failed to register Optimized Face Detection API: {e}")

# Include Anti-Spoofing API
if anti_spoofing_router is not None:
    app.include_router(anti_spoofing_router, tags=["Anti-Spoofing"])
    logger.info("✅ Anti-spoofing API endpoints registered")
else:
    logger.warning("⚠️ Anti-spoofing API not available")

# Include Age-Gender API
if age_gender_router is not None:
    app.include_router(age_gender_router, tags=["Age-Gender"])
    logger.info("✅ Age-gender API endpoints registered")
else:
    logger.warning("⚠️ Age-gender API not available")

# Include Enhanced Face Validation API
try:
    from .api.face_enhanced_validation import router as enhanced_validation_router
    app.include_router(enhanced_validation_router, tags=["Enhanced Face Validation"])
    logger.info("✅ Enhanced Face Validation API endpoints registered")
except ImportError as e:
    logger.warning(f"⚠️ Enhanced Face Validation API not available: {e}")
except Exception as e:
    logger.error(f"❌ Failed to register Enhanced Face Validation API: {e}")

# Add health endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for Docker container monitoring"""
    try:
        # Basic health check
        services_status = {}
        
        # Check if services are initialized
        face_detection_service = getattr(app.state, FACE_DETECTION_SERVICE, None)
        face_recognition_service = getattr(app.state, FACE_RECOGNITION_SERVICE, None)
        face_analysis_service = getattr(app.state, FACE_ANALYSIS_SERVICE, None)
        vram_manager = getattr(app.state, VRAM_MANAGER_SERVICE, None)
        
        services_status["face_detection"] = face_detection_service is not None
        services_status["face_recognition"] = face_recognition_service is not None  
        services_status["face_analysis"] = face_analysis_service is not None
        services_status["vram_manager"] = vram_manager is not None
        
        # Check if all critical services are running
        all_services_healthy = all([
            services_status["face_detection"],
            services_status["face_recognition"],
            services_status["vram_manager"]
        ])
        
        return {
            "status": "healthy" if all_services_healthy else "degraded",
            "services": services_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

# Root endpoint redirect
@app.get("/")
async def root():
    """Redirect root to API documentation"""
    return RedirectResponse(url="/docs")

@app.get("/health", tags=["System"])
async def health_check(request: Request) -> Dict[str, Any]:
    """Perform a health check of the system and its services."""
    service_statuses: Dict[str, Dict[str, Any]] = {}
    overall_healthy = True
    total_services = 0
    active_services = 0

    for service_key in USER_FACING_SERVICE_KEYS:
        total_services += 1
        service_instance = getattr(request.app.state, service_key, None)
        if service_instance and hasattr(service_instance, 'get_service_info'):
            try:
                # ตรวจสอบว่าเป็น async method หรือไม่
                import inspect
                if inspect.iscoroutinefunction(service_instance.get_service_info):
                    info = await service_instance.get_service_info()
                else:
                    info = service_instance.get_service_info()
                
                # ตรวจสอบว่า info เป็น dict หรือไม่
                if isinstance(info, dict):
                    service_statuses[service_key] = {
                        "status": "healthy",
                        "details": info
                    }
                    active_services += 1
                else:
                    service_statuses[service_key] = {
                        "status": "warning",
                        "details": {"raw_response": str(info)},
                        "message": "Service returned non-dict response"
                    }
                    active_services += 1
                    
            except Exception as e:
                logger.error(f"Error getting info for {service_key}: {e}")
                service_statuses[service_key] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                overall_healthy = False
        else:
            service_statuses[service_key] = {
                "status": "unavailable",
                "error": "Service not found or does not support get_service_info"
            }
            overall_healthy = False
            
    # Check VRAM manager separately
    vram_manager = getattr(request.app.state, VRAM_MANAGER_SERVICE, None)
    if vram_manager and hasattr(vram_manager, 'get_vram_status'):
        try:
            vram_status = await vram_manager.get_vram_status()
            service_statuses[VRAM_MANAGER_SERVICE] = {
                "status": "healthy",
                "details": vram_status
            }
        except Exception as e:
            logger.error(f"Error getting VRAM status: {e}")
            service_statuses[VRAM_MANAGER_SERVICE] = {
                "status": "unhealthy",
                "error": str(e)
            }
    elif not vram_manager:
        service_statuses[VRAM_MANAGER_SERVICE] = {
            "status": "unavailable",
            "error": "VRAM Manager not found"
        }
        overall_healthy = False

    return {
        "status": "healthy" if overall_healthy else "degraded",
        "total_services": total_services,
        "active_services": active_services,
        "service_details": service_statuses,
        "system_info": {
            "python_version": sys.version,
            "platform": sys.platform,
            "project_root": str(project_root)
        }
    }

# Main entry point
if __name__ == "__main__":
    settings = get_settings()
    logger.info(f"Starting server on {settings.host}:{settings.port}")
    
    # Ensure log directory exists  
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )