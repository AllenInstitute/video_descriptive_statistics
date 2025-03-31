import numpy as np
import zarr
import json
import os
import utils
import pickle
import utils
from pathlib import Path

class VideoStats():
    def __init__(self, frame_zarr_path: Path):
        self.frame_zarr_path = frame_zarr_path
        
        self.number_of_frames_to_use = 100

    def _load_metadata(self):
            """Load metadata from the Zarr store."""
            zarr_store_frames = self._get_zarr_store_frame()
            root_group = zarr.open_group(zarr_store_frames, mode='r')
            metadata = json.loads(root_group.attrs['metadata'])
            self.video_metadata = metadata
            return self

    def _get_zarr_store_frame(self):
        zarr_store_frames = zarr.DirectoryStore(self.frame_zarr_path)
        return zarr_store_frames

    def _compute_contrast(self, frames):
        contrast_array = []
        for frame in frames:
            contrast = np.divide(np.max(frame)-np.min(frame),np.max(frame)+np.min(frame))
            contrast_array.append(contrast)
        self.contrast_array = contrast_array
        return self

        
    def _compute_mtf(self, frame):
        """
        Compute the Modulation Transfer Function (MTF) of a cropped region in an frame.
        
        Args:
            frame (numpy.ndarray): The input frame.
        
        Returns:
            numpy.ndarray: The computed MTF of the cropped region.
        """
        
        # Compute the Fourier Transform
        f = fft2(cropped_frame)
        fshift = fftshift(f)
        magnitude_spectrum = np.abs(fshift)
        
        # Compute the MTF
        mtf = magnitude_spectrum / np.max(magnitude_spectrum)
        self.modulation_transfer_function = mtf
        
        return self


    def _compute_snr(self, frames):
        """
        Compute the Signal-to-Noise Ratio (SNR) of an frame or a cropped region.
        
        Args:
            frame (numpy.ndarray): The input frame.
        
        Returns:
            array: The computed SNR of the frame or cropped region.
        """
        snr_array = []
        for frame in frames:
            
            # Compute the SNR
            snr = np.mean(frame) / np.std(frame)
            snr_array.append(snr)

        self.snr_array = snr_array
        return self
        
            