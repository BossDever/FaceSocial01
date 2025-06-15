#!/usr/bin/env python3
"""
Multi-Framework Face Recognition Testing Script
สคริปต์ทดสอบระบบจดจำใบหน้าแบบหลายเฟรมเวิร์ก

Usage:
    python test_multi_framework.py --test-type [single|batch|compare|benchmark]
    
Examples:
    python test_multi_framework.py --test-type single --image test_images/boss_01.jpg
    python test_multi_framework.py --test-type compare --test-dir test_images/
    python test_multi_framework.py --test-type benchmark --iterations 100
"""

import asyncio
import argparse
import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any
import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import after adding to path
try:
    from src.ai_services.face_recognition.face_recognition_service import FaceRecognitionService
    from src.core.log_config import get_logger
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running this script from the project root directory.")
    sys.exit(1)

logger = get_logger(__name__)

class MultiFrameworkTester:
    """Multi-framework face recognition tester"""
    
    def __init__(self, models_path: str = "./model/face-recognition/"):
        self.models_path = models_path
        self.service = None
        
    async def initialize_service(self):
        """Initialize face recognition service with multi-framework support"""
        try:
            logger.info("🔧 Initializing Multi-Framework Face Recognition Service...")
            
            config = {
                "preferred_model": "facenet",
                "similarity_threshold": 0.50,
                "unknown_threshold": 0.40,
                "enable_gpu_optimization": True,
                "use_case": "general_purpose"
            }
            
            self.service = FaceRecognitionService(
                config=config,
                enable_multi_framework=True,
                frameworks=None,  # Auto-detect all available
                models_path=self.models_path
            )
            
            # Initialize service
            await self.service.initialize()
            
            logger.info("✅ Service initialized successfully!")
            logger.info(f"Available frameworks: {self.service.get_available_frameworks()}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Service initialization failed: {e}")
            return False
    
    async def test_single_recognition(self, image_path: str):
        """Test single image recognition with all frameworks"""
        logger.info(f"🔍 Testing single recognition: {image_path}")
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"❌ Cannot load image: {image_path}")
                return
            
            # Test multi-framework recognition
            result = self.service.recognize_multi_framework(
                image, 
                return_all_results=True
            )
            
            if result.get("success"):
                logger.info("✅ Recognition Results:")
                logger.info(f"   Consensus: {result.get('consensus_prediction', 'Unknown')}")
                logger.info(f"   Confidence: {result.get('confidence', 0.0):.3f}")
                logger.info(f"   Processing Time: {result.get('processing_time', 0.0):.3f}s")
                
                # Show framework-specific results
                framework_results = result.get("framework_results", {})
                for framework, fw_result in framework_results.items():
                    logger.info(f"   {framework}: {fw_result['person_id']} ({fw_result['confidence']:.3f})")
            else:
                logger.error(f"❌ Recognition failed: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"❌ Single recognition test failed: {e}")
    
    async def test_batch_recognition(self, test_dir: str):
        """Test batch recognition on multiple images"""
        logger.info(f"📁 Testing batch recognition: {test_dir}")
        
        try:
            # Get all image files
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_files.extend(Path(test_dir).glob(f"*{ext}"))
                image_files.extend(Path(test_dir).glob(f"*{ext.upper()}"))
            
            if not image_files:
                logger.warning(f"⚠️ No image files found in {test_dir}")
                return
            
            logger.info(f"Found {len(image_files)} images")
            
            # Load images
            images = []
            valid_files = []
            
            for img_file in image_files:
                image = cv2.imread(str(img_file))
                if image is not None:
                    images.append(image)
                    valid_files.append(img_file.name)
            
            logger.info(f"Loaded {len(images)} valid images")
            
            # Run batch recognition
            start_time = time.time()
            
            batch_result = self.service.batch_recognize_multi_framework(
                images=images,
                max_concurrent=4
            )
            
            total_time = time.time() - start_time
            
            if batch_result.get("success"):
                logger.info("✅ Batch Recognition Results:")
                logger.info(f"   Total Images: {batch_result.get('total_images', 0)}")
                logger.info(f"   Processed: {batch_result.get('processed_images', 0)}")
                logger.info(f"   Total Time: {total_time:.3f}s")
                logger.info(f"   Avg Time/Image: {batch_result.get('average_time_per_image', 0.0):.3f}s")
                
                # Show some results
                batch_results = batch_result.get("batch_results", [])
                for i, result in enumerate(batch_results[:5]):  # Show first 5
                    if 'results' in result:
                        logger.info(f"   Image {i+1} ({valid_files[i]}): Multiple predictions")
                    else:
                        logger.info(f"   Image {i+1} ({valid_files[i]}): {result.get('error', 'Unknown error')}")
            else:
                logger.error(f"❌ Batch recognition failed: {batch_result.get('error')}")
                
        except Exception as e:
            logger.error(f"❌ Batch recognition test failed: {e}")
    
    async def test_framework_comparison(self, test_dir: str):
        """Test framework comparison with ground truth"""
        logger.info(f"⚖️ Testing framework comparison: {test_dir}")
        
        try:
            # Create test data with ground truth
            test_images = []
            
            # Look for images with person names in filename
            for img_file in Path(test_dir).glob("*.*"):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image = cv2.imread(str(img_file))
                    if image is not None:
                        # Extract person name from filename
                        # Example: boss_01.jpg -> boss
                        person_name = img_file.stem.split('_')[0] if '_' in img_file.stem else img_file.stem
                        test_images.append((image, person_name))
            
            if not test_images:
                logger.warning(f"⚠️ No test images found in {test_dir}")
                return
            
            logger.info(f"Prepared {len(test_images)} test images")
            
            # Run comparison
            comparison_result = self.service.compare_frameworks(
                test_images=test_images,
                output_dir="./output/framework_comparison/"
            )
            
            if comparison_result.get("success"):
                logger.info("✅ Framework Comparison Completed!")
                
                results = comparison_result.get("results", {})
                performance_metrics = results.get("performance_metrics", {})
                
                logger.info("📊 Performance Summary:")
                for framework, metrics in performance_metrics.items():
                    logger.info(f"   {framework}:")
                    logger.info(f"     Accuracy: {metrics.get('accuracy', 0.0):.3f}")
                    logger.info(f"     Avg Time: {metrics.get('avg_processing_time', 0.0):.3f}s")
                    logger.info(f"     Avg Confidence: {metrics.get('avg_confidence', 0.0):.3f}")
                
                # Save detailed results
                output_file = "./output/framework_comparison/detailed_results.json"
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                
                logger.info(f"💾 Detailed results saved to {output_file}")
            else:
                logger.error(f"❌ Framework comparison failed: {comparison_result.get('error')}")
                
        except Exception as e:
            logger.error(f"❌ Framework comparison test failed: {e}")
    
    async def test_speed_benchmark(self, iterations: int = 100):
        """Test processing speed benchmark"""
        logger.info(f"⏱️ Testing speed benchmark ({iterations} iterations)")
        
        try:
            # Create dummy test image
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # Run benchmark
            benchmark_result = self.service.benchmark_frameworks(
                test_image=test_image,
                iterations=iterations
            )
            
            if benchmark_result.get("success"):
                logger.info("✅ Speed Benchmark Results:")
                
                results = benchmark_result.get("benchmark_results", {})
                
                # Sort by speed (FPS)
                sorted_frameworks = sorted(
                    results.items(),
                    key=lambda x: x[1].get('fps', 0),
                    reverse=True
                )
                
                logger.info("🏆 Framework Speed Ranking:")
                for rank, (framework, metrics) in enumerate(sorted_frameworks, 1):
                    avg_time = metrics.get('avg_time', 0.0)
                    fps = metrics.get('fps', 0.0)
                    logger.info(f"   {rank}. {framework}: {avg_time:.3f}s ({fps:.1f} FPS)")
                
                # Save benchmark results
                output_file = "./output/benchmark_results.json"
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                
                logger.info(f"💾 Benchmark results saved to {output_file}")
            else:
                logger.error(f"❌ Speed benchmark failed: {benchmark_result.get('error')}")
                
        except Exception as e:
            logger.error(f"❌ Speed benchmark test failed: {e}")
    
    async def add_test_persons(self):
        """Add test persons to the gallery"""
        logger.info("👥 Adding test persons to gallery...")
        
        try:
            test_persons = [
                {
                    "person_id": "boss",
                    "person_name": "Boss Person",
                    "images": ["test_images/boss_01.jpg", "test_images/boss_02.jpg", "test_images/boss_03.jpg"]
                },
                {
                    "person_id": "night",
                    "person_name": "Night Person", 
                    "images": ["test_images/night_01.jpg", "test_images/night_02.jpg", "test_images/night_03.jpg"]
                }
            ]
            
            # Load and add persons
            for person_data in test_persons:
                images = []
                for img_path in person_data["images"]:
                    if os.path.exists(img_path):
                        image = cv2.imread(img_path)
                        if image is not None:
                            images.append(image)
                
                if images:
                    result = self.service.add_face_multi_framework(
                        person_id=person_data["person_id"],
                        person_name=person_data["person_name"],
                        face_images=images
                    )
                    
                    if result.get("success"):
                        logger.info(f"✅ Added {person_data['person_name']} with {len(images)} images")
                    else:
                        logger.error(f"❌ Failed to add {person_data['person_name']}: {result.get('error')}")
                else:
                    logger.warning(f"⚠️ No valid images found for {person_data['person_name']}")
                    
        except Exception as e:
            logger.error(f"❌ Failed to add test persons: {e}")

async def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Multi-Framework Face Recognition Tester")
    parser.add_argument("--test-type", choices=["single", "batch", "compare", "benchmark", "all"], 
                       default="all", help="Type of test to run")
    parser.add_argument("--image", help="Image path for single recognition test")
    parser.add_argument("--test-dir", default="test_images/", help="Directory with test images")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations for benchmark")
    parser.add_argument("--models-path", default="./model/face-recognition/", help="Path to model files")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = MultiFrameworkTester(models_path=args.models_path)
    
    # Initialize service
    if not await tester.initialize_service():
        logger.error("❌ Failed to initialize service. Exiting.")
        return 1
    
    # Add test persons first
    await tester.add_test_persons()
    
    try:
        if args.test_type == "single" or args.test_type == "all":
            image_path = args.image or "test_images/boss_01.jpg"
            if os.path.exists(image_path):
                await tester.test_single_recognition(image_path)
            else:
                logger.warning(f"⚠️ Image not found: {image_path}")
        
        if args.test_type == "batch" or args.test_type == "all":
            if os.path.exists(args.test_dir):
                await tester.test_batch_recognition(args.test_dir)
            else:
                logger.warning(f"⚠️ Test directory not found: {args.test_dir}")
        
        if args.test_type == "compare" or args.test_type == "all":
            if os.path.exists(args.test_dir):
                await tester.test_framework_comparison(args.test_dir)
            else:
                logger.warning(f"⚠️ Test directory not found: {args.test_dir}")
        
        if args.test_type == "benchmark" or args.test_type == "all":
            await tester.test_speed_benchmark(args.iterations)
        
        logger.info("🎉 All tests completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("⏹️ Tests interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        return 1
    finally:
        # Cleanup
        if tester.service:
            await tester.service.shutdown()

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️ Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
