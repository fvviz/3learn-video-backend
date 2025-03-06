import pathlib
import sys
import subprocess
import PIL.Image
from typing import List, Union
from google import genai

def analyze_student_attention(images: List[Union[str, PIL.Image.Image]], api_key: str, custom_prompt: str = None) -> str:
    """
    Analyzes multiple student webcam images to assess attention levels.
    
    Args:
        images: List of either image paths (str) or PIL Image objects
        api_key: Gemini API key
        custom_prompt: Optional custom prompt for final analysis
    
    Returns:
        str: Analysis response from Gemini in a structured format
    """
    client = genai.Client(api_key=api_key)
    
    # If custom prompt is provided, use it for final analysis
    if custom_prompt:
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[custom_prompt]
        )
        return response.text
    
    # Convert any string paths to PIL Images
    processed_images = []
    for img in images:
        if isinstance(img, str):
            processed_images.append(PIL.Image.open(img))
        else:
            processed_images.append(img)
    
    # Enhanced prompt for analysis
    contents = [
        """
        You are an expert supervisor monitoring student attention in an online class through webcam screenshots.
        Analyze the student's attention levels and behavior in detail. DO not respond with anything but the final analysis.
        
        Provide your analysis in the following structured format:
        
        1. ATTENTIVENESS_RATING (1-10): Give an overall rating
        
        2. EYE_CONTACT_SCORE (1-10): Rate how well the student maintains eye contact with the screen
        - Consider: gaze direction, frequency of looking away
        
        3. POSTURE_SCORE (1-10): Evaluate the student's sitting posture
        - Consider: upright position, slouching, distance from screen
        
        4. FOCUS_DURATION: Estimate the percentage of time the student appears focused
        
        5. DETAILED_OBSERVATIONS:
        - List specific behaviors observed
        - Note any distractions
        - Describe engagement indicators
        
        Format each metric clearly with "METRIC: score" on its own line for easy parsing.
        You will directly return your result only.
        """
    ]
    contents.extend(processed_images)
    
    # Generate content using the images
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=contents
    )
    
    return response.text



    

