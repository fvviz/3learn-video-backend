import gradio as gr
import cv2
import time
from datetime import datetime
import os
import requests
import threading
import random

# FastAPI server configuration
SERVER_URL = "http://localhost:8000"

# Create directory for saving images if it doesn't exist
if not os.path.exists('webcam_captures'):
    os.makedirs('webcam_captures')

class AttentionMonitor:
    def __init__(self):
        self.job_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_status = "Session not started"
        self.is_capturing = False
        self.capture_thread = None
        self.current_batch_images = []
        
        # Start background status update thread
        self.status_thread = threading.Thread(target=self.fetch_status_periodically, daemon=True)
        self.status_thread.start()
        
    def create_job(self):
        """Create a job on the server"""
        try:
            response = requests.post(f"{SERVER_URL}/create_job", json={"job_id": self.job_id})
            if response.status_code == 200:
                print("Job created successfully.")
                self.current_status = "Job started"
            else:
                print(f"Error creating job: {response.text}")
        except Exception as e:
            print(f"Error creating job: {str(e)}")

    def fetch_status_periodically(self):
        """Fetch job status every 5 seconds and update UI"""
        while True:
            time.sleep(5)
            try:
                response = requests.post(f"{SERVER_URL}/job_status", json={"job_id": self.job_id})
                if response.status_code == 200:
                    self.current_status = response.json()["status"]
            except Exception as e:
                print(f"Error fetching status: {str(e)}")

    def capture_images(self):
        """Capture images every second for 10 seconds"""
        cap = cv2.VideoCapture(0)
        captured_frames = []
        for _ in range(10):
            ret, frame = cap.read()
            if ret:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f'webcam_captures/frame_{timestamp}.jpg'
                cv2.imwrite(filename, frame)
                captured_frames.append(filename)
            time.sleep(1)
        cap.release()
        
        # Select 5-6 random frames for analysis
        selected_images = random.sample(captured_frames, random.randint(5, 6))
        self.analyze_images(selected_images)

    def analyze_images(self, image_paths):
        """Send selected images to the server for analysis"""
        try:
            response = requests.post(f"{SERVER_URL}/analyze_student_images", json={"job_id": self.job_id, "image_paths": image_paths})
            print(response.json())
            for path in image_paths:
                os.remove(path)
        except Exception as e:
            print(f"Error during analysis request: {e}")

    def get_session_summary(self):
        """Get the final analysis summary from the server"""
        try:
            response = requests.post(f"{SERVER_URL}/analyze_job", json={"job_id": self.job_id})
            if response.status_code == 200:
                return response.json()
            else:
                return f"Error getting session summary: {response.text}"
        except Exception as e:
            return f"Error getting final analysis: {str(e)}"

def create_ui():
    monitor = AttentionMonitor()
    monitor.create_job()
    
    with gr.Blocks() as app:
        gr.Markdown("# Student Attention Monitor")
        with gr.Row():
            with gr.Column():
                webcam = gr.Video(label="Webcam Preview")
                capture_btn = gr.Button("Start Capturing")
            with gr.Column():
                status_output = gr.Textbox(label="Job Status", interactive=False, lines=10)
        
        analyze_btn = gr.Button("Analyze Session")
        analysis_output = gr.Textbox(label="Session Analysis", interactive=False, lines=10)
        
        def start_capture():
            monitor.capture_images()
            return "Capturing images..."
        
        def analyze_session():
            return monitor.get_session_summary()
        
        def update_status():
            return gr.update(value=monitor.current_status)  # Dynamically update status
        
        capture_btn.click(start_capture)
        analyze_btn.click(analyze_session, outputs=[analysis_output])
        
        # **Fix: Periodic status update without `every=10`**
        def periodic_ui_update():
            while True:
                time.sleep(5)
                status_output.update(value=monitor.current_status)

        threading.Thread(target=periodic_ui_update, daemon=True).start()
    
    return app

if __name__ == "__main__":
    app = create_ui()
    app.launch()
