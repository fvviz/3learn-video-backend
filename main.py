import cv2
import time
from datetime import datetime
import os
from typing import List
import random
import threading
import requests
import json

# FastAPI server configuration
SERVER_URL = "http://localhost:8000"

# Create directory for saving images if it doesn't exist
if not os.path.exists('webcam_captures'):
    os.makedirs('webcam_captures')

# Initialize the webcam
cap = cv2.VideoCapture(1)  # Changed to 0 for default webcam

def analyze_images(job_id: str, image_paths: List[str]):
    """Send images to server for analysis"""
    try:
        response = requests.post(
            f"{SERVER_URL}/analyze_student_images",
            json={
                "job_id": job_id,
                "image_paths": image_paths
            }
        )
        if response.status_code == 200:
            result = response.json()
            print(f"\nAnalysis Request Status: {result['status']}")
            if 'queue_position' in result:
                print(f"Queue Position: {result['queue_position']}")
            print(f"Message: {result['message']}")
        else:
            print(f"Error analyzing images: {response.text}")
    except Exception as e:
        print(f"Error during analysis request: {e}")

def delete_images(image_paths: List[str]):
    """Delete processed images"""
    for path in image_paths:
        try:
            os.remove(path)
            print(f"Deleted: {path}")
        except Exception as e:
            print(f"Error deleting {path}: {e}")

def process_attention(job_id: str, images_to_process: List[str]):
    """Process a batch of images"""
    try:
        # Send all images for analysis
        analyze_images(job_id, images_to_process)
        
        # Delete all images from the batch
        delete_images(images_to_process)
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")

def wait_for_processing(job_id: str, max_retries: int = 30) -> bool:
    """Wait for all image processing to complete"""
    print("\nWaiting for processing to complete...")
    for i in range(max_retries):
        try:
            response = requests.post(
                f"{SERVER_URL}/analyze_job",
                json={"job_id": job_id}
            )
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and "metrics" in data:
                    # If we have any entries, processing has started
                    if data["metrics"]["total_entries"] > 0:
                        print(f"Processing complete. Found {data['metrics']['total_entries']} entries.")
                        return True
            elif response.status_code == 404:
                print(f"Waiting for first batch to process... ({i+1}/{max_retries})")
            else:
                print(f"Unexpected response: {response.status_code}")
            
            # Exponential backoff: wait longer as time goes on
            wait_time = min(2 * (i + 1), 10)  # Start at 2s, max 10s
            time.sleep(wait_time)
            
        except Exception as e:
            print(f"Error while waiting: {str(e)}")
            time.sleep(2)
    return False

def get_session_summary(job_id: str):
    """Get the final analysis summary from the server"""
    if not wait_for_processing(job_id):
        print("\nTimeout waiting for processing to complete. Attempting final summary anyway...")
    
    try:
        response = requests.post(
            f"{SERVER_URL}/analyze_job",
            json={"job_id": job_id}
        )
        if response.status_code == 200:
            summary = response.json()
            
            if not summary.get("metrics", {}).get("total_entries", 0):
                print("\nNo data was processed for this session.")
                return
            
            print("\n" + "="*50)
            print("SESSION SUMMARY")
            print("="*50)
            
            # Display Metrics
            metrics = summary["metrics"]
            print("\nMETRICS:")
            print(f"Total Entries: {metrics['total_entries']}")
            print(f"Average Attentiveness: {metrics['average_attentiveness']:.2f}/10")
            print(f"Average Eye Contact: {metrics['average_eye_contact']:.2f}/10")
            print(f"Average Posture: {metrics['average_posture']:.2f}/10")
            print(f"Total Focus Duration: {metrics['total_focus_duration']} seconds")
            
            # Display Analysis
            if "analysis" in summary:
                print("\nDETAILED ANALYSIS:")
                print(summary["analysis"])
            
            print("="*50)
            
        else:
            print(f"\nError getting session summary: {response.text}")
    except Exception as e:
        print(f"\nError getting final analysis: {str(e)}")

def main():
    try:
        # Create a unique job ID using timestamp
        job_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Started monitoring session with job ID: {job_id}")
        
        last_capture_time = time.time()
        capture_interval = 5  # Capture every 5 seconds
        current_batch_images = []
        analysis_lock = threading.Lock()
        processing_threads = []

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Show the frame
            cv2.imshow('Webcam Feed (Press q to quit)', frame)

            current_time = time.time()
            
            # Capture frame every interval
            if current_time - last_capture_time >= capture_interval:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f'webcam_captures/frame_{timestamp}.jpg'
                cv2.imwrite(filename, frame)
                
                with analysis_lock:
                    current_batch_images.append(filename)
                    # When we have 5 images, process them
                    if len(current_batch_images) >= 5:
                        images_to_process = current_batch_images.copy()
                        current_batch_images = []
                        thread = threading.Thread(
                            target=process_attention,
                            args=(job_id, images_to_process),
                            daemon=True
                        )
                        thread.start()
                        processing_threads.append(thread)
                
                last_capture_time = current_time
                print(f"Captured frame: {filename}")

            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Process any remaining images
        with analysis_lock:
            if current_batch_images:
                thread = threading.Thread(
                    target=process_attention,
                    args=(job_id, current_batch_images),
                    daemon=True
                )
                thread.start()
                processing_threads.append(thread)

        # Wait for all processing threads to complete
        print("\nWaiting for processing threads to complete...")
        for thread in processing_threads:
            thread.join(timeout=30)  # Wait up to 30 seconds per thread
            
        # Additional wait to ensure server processing is complete
        time.sleep(5)  # Give server a moment to finish processing

        # Get and display the final session summary
        get_session_summary(job_id)

    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
