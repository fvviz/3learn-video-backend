import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import csv
from datetime import datetime
import pandas as pd
from gemini_analysis import analyze_student_attention
import requests
from PIL import Image
from io import BytesIO
from collections import defaultdict
import asyncio
from asyncio import create_task

app = FastAPI()

# CORS middleware setu
# Constants and global state
API_KEY = "AIzaSyBEXbKEkhwdvpf62CAMMVymn-MJ1Tt3Sjg"
LOG_DIR = "log_files"
os.makedirs(LOG_DIR, exist_ok=True)

# Job state management
active_jobs = {}  # Track if a job is currently processing
job_queues = defaultdict(asyncio.Queue)  # Queue for each job
job_queue_counts = defaultdict(int)  # Track number of requests in queue for each job

# Request models
class CreateJobRequest(BaseModel):
    job_id: str

class AnalyzeImagesRequest(BaseModel):
    job_id: str
    image_paths: Optional[List[str]] = None
    image_urls: Optional[List[str]] = None

class AnalyzeJobRequest(BaseModel):
    job_id: str

def get_csv_path(job_id: str) -> str:
    return os.path.join(LOG_DIR, f"{job_id}.csv")

def create_csv_file(job_id: str):
    csv_path = get_csv_path(job_id)
    headers = ['timestamp', 'attentiveness_rating', 'comment', 'eye_contact_score', 'posture_score', 'focus_duration']
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

@app.post("/create_job")
async def create_job(request: CreateJobRequest):
    csv_path = get_csv_path(request.job_id)
    
    if os.path.exists(csv_path):
        raise HTTPException(status_code=400, detail="Job ID already exists")
    
    create_csv_file(request.job_id)
    return {"message": f"Job {request.job_id} created successfully"}

async def process_images(job_id: str, images: List[Image.Image]):
    """Process images and update CSV file"""
    try:
        # Get analysis from Gemini
        analysis = analyze_student_attention(images, API_KEY)
        
        # Extract metrics from analysis
        metrics = {
            'rating': 5.0,
            'eye_contact_score': 5.0,
            'posture_score': 5.0,
            'focus_duration': 30
        }
        
        # Parse the analysis text for metrics
        lines = analysis.split('\n')
        for line in lines:
            line = line.strip()
            
            try:
                if 'ATTENTIVENESS_RATING' in line:
                    # Format expected: "METRIC: ATTENTIVENESS_RATING: 7"
                    value = line.split(':')[-1].strip()  # Get the last part after ":"
                    metrics['rating'] = float(value)
                
                elif 'EYE_CONTACT_SCORE' in line:
                    value = line.split(':')[-1].strip()
                    metrics['eye_contact_score'] = float(value)
                
                elif 'POSTURE_SCORE' in line:
                    value = line.split(':')[-1].strip()
                    metrics['posture_score'] = float(value)
                
                elif 'FOCUS_DURATION' in line:
                    value = line.split(':')[-1].strip()
                    percentage = float(value.replace('%', ''))
                    metrics['focus_duration'] = int((percentage / 100) * 60)
            
            except (IndexError, ValueError) as e:
                print(f"Error parsing line '{line}': {str(e)}")
                continue

        # Write to CSV
        csv_path = get_csv_path(job_id)
        with open(csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.now().isoformat(),
                metrics['rating'],
                analysis,
                metrics['eye_contact_score'],
                metrics['posture_score'],
                metrics['focus_duration']
            ])

    finally:
        active_jobs[job_id] = False

@app.post("/analyze_student_images")
async def analyze_student_images(request: AnalyzeImagesRequest, background_tasks: BackgroundTasks):
    job_id = request.job_id
    csv_path = get_csv_path(job_id)
    
    # If job doesn't exist, create it
    if not os.path.exists(csv_path):
        try:
            create_csv_file(job_id)
            print(f"Created new job: {job_id}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create job: {str(e)}")
    
    # Queue management - if job is active or queue is not empty
    if active_jobs.get(job_id) or not job_queues[job_id].empty():
        # Increment queue count before adding to queue
        job_queue_counts[job_id] += 1
        await job_queues[job_id].put(request)
        return {
            "status": "queued", 
            "message": f"Job {job_id} is queued for processing",
            "queue_position": job_queue_counts[job_id]
        }

    # First request - initialize and start processing
    active_jobs[job_id] = True
    job_queue_counts[job_id] = 0

    async def process_request(request: AnalyzeImagesRequest):
        try:
            # Process first request
            await process_single_request(request, job_id)
            
            # Process queued requests
            while not job_queues[job_id].empty():
                next_request = await job_queues[job_id].get()
                await process_single_request(next_request, job_id)
                job_queue_counts[job_id] = max(0, job_queue_counts[job_id] - 1)

        finally:
            active_jobs[job_id] = False

    # Start background processing
    background_tasks.add_task(process_request, request)
    
    return {
        "status": "processing", 
        "message": f"Processing started for job {job_id}",
        "queue_count": job_queue_counts[job_id]
    }

async def process_single_request(request: AnalyzeImagesRequest, job_id: str):
    """Process a single request's images"""
    images = []
    
    try:
        if request.image_urls:
            for url in request.image_urls:
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    img = Image.open(BytesIO(response.content))
                    images.append(img)
                except Exception as e:
                    print(f"Error processing URL {url}: {str(e)}")
                    continue
        
        if request.image_paths:
            for path in request.image_paths:
                try:
                    if os.path.exists(path):
                        img = Image.open(path)
                        images.append(img)
                except Exception as e:
                    print(f"Error processing path {path}: {str(e)}")
                    continue

        if images:
            await process_images(job_id, images)

    except Exception as e:
        print(f"Error processing request: {str(e)}")
    
    finally:
        # Only set active_jobs to False when queue is empty
        if job_queues[job_id].empty():
            active_jobs[job_id] = False

@app.post("/analyze_job")
async def analyze_job(request: AnalyzeJobRequest):
    csv_path = get_csv_path(request.job_id)
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return {"message": "No data recorded for this job"}

        # Calculate basic metrics
        metrics = {
            "total_entries": len(df),
            "average_attentiveness": float(df['attentiveness_rating'].mean()),
            "average_eye_contact": float(df['eye_contact_score'].mean()),
            "average_posture": float(df['posture_score'].mean()),
            "total_focus_duration": int(df['focus_duration'].sum()),
            "latest_comment": str(df['comment'].iloc[-1])
        }

        # Create a prompt for final analysis
        all_comments = "\n".join(df['comment'].tolist())
        summary_prompt = f"""
        Analyze the following session metrics and provide a comprehensive summary:

        Session Statistics:
        - Total Snapshots: {metrics['total_entries']}
        - Average Attentiveness: {metrics['average_attentiveness']:.2f}/10
        - Average Eye Contact: {metrics['average_eye_contact']:.2f}/10
        - Average Posture: {metrics['average_posture']:.2f}/10
        - Total Focus Duration: {metrics['total_focus_duration']} seconds

        Individual Analyses:
        {all_comments}

        Please provide a structured analysis with the following sections:
        1. OVERALL_SUMMARY: A brief overview of the student's performance
        2. NEGATIVE_OBSERVATIONS: List key negative behaviors and patterns  
        3. AREAS_FOR_IMPROVEMENT: List specific areas needing attention
        4. RECOMMENDATIONS: Practical suggestions for improvement
        5. ENGAGEMENT_PATTERN: Analysis of attention patterns over time
        """

        # Get final analysis from Gemini
        final_analysis = analyze_student_attention([], API_KEY, custom_prompt=summary_prompt)

        return {
            "metrics": metrics,
            "analysis": final_analysis,
            "raw_data": {
                "total_snapshots": len(df),
                "timestamps": df['timestamp'].tolist(),
                "attentiveness_scores": df['attentiveness_rating'].tolist(),
                "eye_contact_scores": df['eye_contact_score'].tolist(),
                "posture_scores": df['posture_score'].tolist(),
                "focus_durations": df['focus_duration'].tolist()
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing job: {str(e)}")

@app.post("/job_status")
async def job_status(request: AnalyzeJobRequest):
    csv_path = get_csv_path(request.job_id)
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return {"message": "No data recorded for this job"}

        # Get the latest entry
        latest = df.iloc[-1]
        
        # Format the timestamp for better readability
        timestamp = datetime.fromisoformat(latest['timestamp'])
        formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")

        return {
            "timestamp": formatted_time,
            "attentiveness_rating": float(latest['attentiveness_rating']),
            "eye_contact_score": float(latest['eye_contact_score']), 
            "posture_score": float(latest['posture_score']),
            "focus_duration": int(latest['focus_duration']),
            "comment": str(latest['comment'])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting job status: {str(e)}")