import requests
import os
import glob
import cv2
import time
from typing import Optional, Dict, Any, cast # Modified import
import json # Added for detailed error printing

API_BASE_URL = "http://127.0.0.1:8080/api"
TEST_IMAGES_DIR = r"D:\\projec-finals\\test_images"
OUTPUT_DIR = r"D:\\projec-finals\\output\\bulk_test_output"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def register_face(image_path: str, person_id: str, person_name: str, model_name: str = "facenet") -> Optional[Dict[str, Any]]:
    """Registers a face using the /api/face-recognition/add-face endpoint."""
    url = f"{API_BASE_URL}/face-recognition/add-face"
    data = {
        'person_name': person_name
    }
    try:
        with open(image_path, 'rb') as f_img:
            files = {'file': f_img}
            print(f"Registering {person_id} with image {os.path.basename(image_path)}...")
            start_time = time.time()
            response = requests.post(url, files=files, data=data)
            response.raise_for_status() # Will raise an HTTPError for 4xx/5xx responses
            end_time = time.time()
            reg_time = end_time - start_time
            result = cast(Dict[str, Any], response.json())
            print(f"Registration for {person_id} ({os.path.basename(image_path)}) SUCCESS: {result.get('message', 'No message')} (took {reg_time:.2f}s)")
            return result
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error registering {person_id} with {os.path.basename(image_path)}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Status Code: {e.response.status_code}")
            try:
                error_json = e.response.json()
                if "detail" in error_json:
                    print("Error Detail:")
                    # Use json.dumps for potentially complex/nested detail structures
                    print(json.dumps(error_json["detail"], indent=2))
                if "body" in error_json: # Our sanitized request body from main.py's error handler
                    print(f"Sanitized Request Body (from error response): {error_json['body']}")
                
                # If 'detail' or 'body' was not in error_json, or if we want to print the full thing anyway
                # and it wasn't printed by the checks above, print the whole JSON.
                # This condition ensures we don't just print e.response.text if detail/body were found.
                if not ("detail" in error_json or "body" in error_json):
                    print(f"Response JSON: {error_json}")

            except ValueError: # If response is not JSON
                print(f"Response content (not JSON): {e.response.text}")
        return None
    # 'finally' block for closing f_img is not needed due to 'with' statement

def analyze_and_draw_faces(image_path: str, output_dir: str, 
                           detection_model: str = "yolov9c", 
                           recognition_model: str = "facenet") -> None:
    """Analyzes an image for faces, recognizes them, draws bounding boxes and labels."""
    url = f"{API_BASE_URL}/face-analysis/analyze"
    
    try:
        with open(image_path, 'rb') as f_img:
            files = {'file': (os.path.basename(image_path), f_img, 'image/jpeg')}
            print(f"Analyzing {os.path.basename(image_path)}...")
            response = requests.post(url, files=files)
            response.raise_for_status()
            analysis_results = response.json()

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path} for analysis.")
        return
    except requests.exceptions.RequestException as e:
        print(f"API Error analyzing {os.path.basename(image_path)}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        return
    except Exception as e: # Catch other errors like JSONDecodeError
        print(f"Unexpected error during API call for {os.path.basename(image_path)}: {e}")
        return

    if not analysis_results:
        print(f"Failed to get analysis results for {os.path.basename(image_path)}.")
        return

    try:
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            print(f"Error: Could not read image {image_path} with OpenCV for drawing.")
            return

        if 'detection_results' in analysis_results and analysis_results['detection_results']:
            detections = analysis_results['detection_results']
            recognition_results_list = analysis_results.get('recognition_results', [])
            
            for i, det in enumerate(detections):
                box = det.get('box')
                if not box or len(box) != 4:
                    print(f"Warning: Invalid box format for a detection in {os.path.basename(image_path)}")
                    continue
                
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = "Unknown"
                rec_data = None
                if i < len(recognition_results_list): # Check if recognition result exists for this detection
                    rec_data = recognition_results_list[i]
                
                if rec_data:
                    best_match = rec_data.get('best_match')
                    if best_match and best_match.get('person_id') != "Unknown" and best_match.get('person_id') is not None:
                        label = best_match.get('person_name', best_match.get('person_id', "Unknown"))
                    elif rec_data.get('error'):
                        label = f"Error: {rec_data.get('error')[:30]}"
                    # If no best_match or person_id is "Unknown", label remains "Unknown"
                
                cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        output_filename = os.path.basename(image_path)
        final_output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(final_output_path, img_cv)
        print(f"Saved annotated image to {final_output_path}")

    except cv2.error as e:
        print(f"OpenCV error processing {os.path.basename(image_path)} after analysis: {e}")
    except Exception as e:
        print(f"Error processing/drawing on {os.path.basename(image_path)} after analysis: {e}")

def main() -> None:
    print("Starting bulk face registration and testing...")

    # --- Registration Phase ---
    boss_images = sorted(glob.glob(os.path.join(TEST_IMAGES_DIR, "boss_[0-9][0-9].jpg")))
    night_images = sorted(glob.glob(os.path.join(TEST_IMAGES_DIR, "night_[0-9][0-9].jpg")))

    print("\n--- Registering Boss ---")
    for img_path in boss_images:
        register_face(img_path, "boss", "Boss")
        time.sleep(0.1) # Small delay between requests

    print("\n--- Registering Night ---")
    for img_path in night_images:
        register_face(img_path, "night", "Night")
        time.sleep(0.1) # Small delay between requests
    
    # --- Analysis Phase ---
    print("\n--- Analyzing all test images ---")
    all_test_images = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg")) + \
                      glob.glob(os.path.join(TEST_IMAGES_DIR, "*.png"))
    
    for img_path in all_test_images:
        analyze_and_draw_faces(img_path, OUTPUT_DIR)
        time.sleep(0.1) # Small delay

    print("\nBulk processing complete.")
    print(f"Annotated images saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
