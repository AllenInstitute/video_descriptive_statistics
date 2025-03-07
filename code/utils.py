import os
import cv2
import numpy as np
import random
import pandas as pd 
from pathlib import Path

matplotlib.rc('ytick', labelsize=20) 

def get_results_folder(pipeline: bool = True) -> Path:
    """
    Get the results folder path.

    Returns:
        str: Path to the results folder.
    """
    if pipeline:
        return Path('/results/')
    else:
        return Path('/root/capsule/results')


def get_data_folder(pipeline: bool = True) -> Path:
    """
    Get the data folder path.

    Returns:
        str: Path to the results folder.
    """
    if pipeline:
        return Path('/data/')
    else:
        return Path('/root/capsule/data')

def object_to_dict(obj):
    if hasattr(obj, "__dict__"):
        return {key: object_to_dict(value) for key, value in vars(obj).items()}
    elif isinstance(obj, list):
        return [object_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: object_to_dict(value) for key, value in obj.items()}
    else:
        return obj

def crop_frame(frame, crop_region):
    x, y, width, height = crop_region
    cropped_frame = frame[y:y+height, x:x+width]
    return cropped_frame



def compute_edges(frame, method='sobel', sobel_ksize=5, canny_threshold1=100, canny_threshold2=200):
    """
    Computes the edges of a frame using the specified edge detection method.
    
    Parameters:
        frame (numpy array): The input image frame (BGR format).
        method (str): The edge detection method to use ('sobel' or 'canny').
        sobel_ksize (int): Kernel size for the Sobel operator.
        canny_threshold1 (int): First threshold for the hysteresis procedure in Canny.
        canny_threshold2 (int): Second threshold for the hysteresis procedure in Canny.
        
    Returns:
        edges (numpy array): The edges detected in the image.
    """
    
    # Blur the frame for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    
    if method == 'sobel':
        # Sobel Edge Detection on both X and Y axes
        edges = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=sobel_ksize)
    elif method == 'canny':
        # Canny Edge Detection
        edges = cv2.Canny(image=img_blur, threshold1=canny_threshold1, threshold2=canny_threshold2)
    else:
        raise ValueError("Invalid method. Use 'sobel' or 'canny'.")
    
    return edges
