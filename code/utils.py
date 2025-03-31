import os
import cv2
import numpy as np
import random
import pandas as pd 
from pathlib import Path


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


def smooth_series(series, window_size):
    """Smooth the series using a moving average.
    
    Args:
        series (pd.Series): The data series to smooth.
        window_size (int): The size of the moving average window.
    
    Returns:
        pd.Series: The smoothed series.
    """
    return series.rolling(window=window_size, min_periods=1).mean()


def normalize_series(series):
    """Normalize the series using min-max scaling.
    
    Args:
        series (pd.Series): The data series to normalize.
    
    Returns:
        pd.Series: The normalized series.
    """
    return (series - series.min()) / (series.max() - series.min())