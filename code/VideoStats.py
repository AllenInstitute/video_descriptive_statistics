import numpy as np
import zarr
import json
import pickle
from pathlib import Path
from scipy.fft import fft2, fftshift  # Required for MTF computation


class VideoStats:
    def __init__(self, frame_zarr_path: Path, number_of_frames_to_use: int = 100):
        """
        Initialize the VideoStats object.

        Args:
            frame_zarr_path (Path): Path to the Zarr file containing video frames.
            number_of_frames_to_use (int): Number of random frames to sample.
        """
        self.frame_zarr_path = frame_zarr_path
        self.number_of_frames_to_use = number_of_frames_to_use

    def _get_zarr_store_frame(self):
        """Return the Zarr store object."""
        return zarr.DirectoryStore(self.frame_zarr_path)

    def _load_metadata(self):
        """Load metadata from the Zarr store."""
        zarr_store_frames = self._get_zarr_store_frame()
        root_group = zarr.open_group(zarr_store_frames, mode='r')
        metadata = json.loads(root_group.attrs['metadata'])
        self.video_metadata = metadata
        return self

    def _load_frames(self):
        """
        Load a random subset of frames from the Zarr store.
        """
        zarr_store_frames = self._get_zarr_store_frame()
        root_group = zarr.open_group(zarr_store_frames, mode='r')
        data = root_group['data']

        n = self.number_of_frames_to_use
        total_frames = data.shape[0]
        random_indices = np.random.choice(total_frames, size=n, replace=False)
        random_frames = np.stack([data[i] for i in random_indices])

        self.frame_indices = random_indices
        self.frames = random_frames
        return self

    def _compute_contrast(self):
        """
        Compute contrast for each frame using the Michelson contrast formula.

        Args:
            frames (np.ndarray): Array of video frames.

        Returns:
            self
        """
        contrast_array = [
            (np.max(frame) - np.min(frame)) / (np.max(frame) + np.min(frame))
            for frame in self.frames
        ]
        self.contrast_array = contrast_array
        return self

    def _compute_mtf(self):
        """
        Compute the Modulation Transfer Function (MTF) of frames.

        Args:
            frame (np.ndarray): Single video frame.

        Returns:
            self
        """
        mtf=[]
        for frame in self.frames:
            f = fft2(frame)
            fshift = fftshift(f)
            magnitude_spectrum = np.abs(fshift)
            mtf.append(magnitude_spectrum / np.max(magnitude_spectrum))
        self.modulation_transfer_function = mtf

        return self

    def _compute_snr(self, frames):
        """
        Compute the Signal-to-Noise Ratio (SNR) for each frame.

        Args:
            frames (np.ndarray): Array of video frames.

        Returns:
            self
        """
        snr_array = []
        for frame in self.frames:
            snr_array.append([np.mean(frame) / np.std(frame) for frame in frames])
        self.snr_array = snr_array
        return self

        
    def _compute_edges(self, method='sobel', sobel_ksize=5, canny_threshold1=100, canny_threshold2=200):
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
        frame_edges = []
        for frame in self.frames:
            # Blur the frame for better edge detection
            img_blur = cv2.GaussianBlur(frame, (3, 3), 0)
                
            if method == 'sobel':
                # Sobel Edge Detection on both X and Y axes
                edges = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=sobel_ksize)
                frame_edges.append(edges)
            elif method == 'canny':
                # Canny Edge Detection
                edges = cv2.Canny(image=img_blur, threshold1=canny_threshold1, threshold2=canny_threshold2)
                frame_edges.append(edges)
            else:
                raise ValueError("Invalid method. Use 'sobel' or 'canny'.")
        
        self.frame_edges = frame_edges
        return self

    def _compute_pixel_distribution(self, bins=256):
        """
        Compute the normalized pixel value distribution across multiple frames.

        Args:
            frames (np.ndarray): Array of frames (N x H x W).
            bins (int): Number of bins for the histogram.

        Returns:
            self
        """
        # Flatten all pixel values
        pixel_values = self.frames.ravel()

        # Compute histogram and normalize
        hist, bin_edges = np.histogram(pixel_values, bins=bins, range=(0, 255), density=True)

        self.pixel_distribution = hist
        self.pixel_bin_edges = bin_edges

        self.pixel_hist_mean = np.mean(pixel_values)
        self.pixel_hist_median = np.median(pixel_values)
        self.pixel_hist_std = np.std(pixel_values)
        self.pixel_hist_skewness = 3 * (self.pixel_hist_mean - self.pixel_hist_median) / self.pixel_hist_std
        return self

    def detect_blur_laplacian(self):
        """
        Detects blur in an frame using the variance of the Laplacian.

        
        Returns:
            laplacian_var (float): Variance of the Laplacian, used to quantify blur.
        """
       
        
        for frame in self.frames:
            # Apply the Laplacian operator to the grayscale frame
            laplacian = cv2.Laplacian(frame, cv2.CV_64F)
            
            # Calculate the variance of the Laplacian
            laplacian_var.append(laplacian.var())
            self.blur_values = laplacian_var
        
        return self
