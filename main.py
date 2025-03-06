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

def get_session_summary(job_id: str):
    """Get the final analysis summary from the server"""
    try:
        response = requests.post(
            f"{SERVER_URL}/analyze_job",
            json={"job_id": job_id}
        )
        if response.status_code == 200:
            summary = response.json()
            
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
            print("\nDETAILED ANALYSIS:")
            print(summary["analysis"])
            
            print("="*50)
            
        else:
            print(f"\nError getting session summary: {response.text}")
    except Exception as e:
        print(f"\nError getting final analysis: {str(e)}")

def get_job_status(job_id: str):
    """Get the current status from the server"""
    try:
        response = requests.post(
            f"{SERVER_URL}/job_status",
            json={"job_id": job_id}
        )
        if response.status_code == 200:
            return response.json()["status"]
        return None
    except Exception as e:
        print(f"Error getting job status: {e}")
        return None

def main():
    try:
        # Create a unique job ID using timestamp
        job_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Started monitoring session with job ID: {job_id}")
        
        last_capture_time = time.time()
        last_status_check = time.time()
        capture_interval = 5  # Capture every 5 seconds
        status_check_interval = 10  # Check status every 10 seconds
        current_batch_images = []
        analysis_lock = threading.Lock()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Show the frame
            cv2.imshow('Webcam Feed (Press q to quit)', frame)

            current_time = time.time()
            
            # Check job status every 10 seconds
            if current_time - last_status_check >= status_check_interval:
                status = get_job_status(job_id)
                if status:
                    print("\nCurrent Student Status:")
                    print("="*50)
                    print(status)
                    print("="*50)
                last_status_check = current_time

            # Capture frame every interval
            if current_time - last_capture_time >= capture_interval:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f'webcam_captures/frame_{timestamp}.jpg'
                cv2.imwrite(filename, frame)
                
                with analysis_lock:
                    current_batch_images.append(filename)
                    if len(current_batch_images) >= 5:
                        images_to_process = current_batch_images.copy()
                        current_batch_images = []
                        threading.Thread(
                            target=process_attention,
                            args=(job_id, images_to_process),
                            daemon=True
                        ).start()
                
                last_capture_time = current_time

            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Process any remaining images
        with analysis_lock:
            if current_batch_images:
                process_attention(job_id, current_batch_images)

        # Wait a bit for processing to complete
        print("\nWaiting for final processing...")
        time.sleep(5)

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
