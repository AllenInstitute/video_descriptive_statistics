import numpy as np
import zarr
import json
import pickle
import cv2
from pathlib import Path
import utils
from scipy.fft import fft2, fftshift

RESULTS_PATH = utils.get_results_folder()#Path("../results")

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
        """Load metadata from the Zarr store and store in self.video_metadata."""
        zarr_store_frames = self._get_zarr_store_frame()
        root_group = zarr.open_group(zarr_store_frames, mode='r')
        metadata = json.loads(root_group.attrs['metadata'])
        self.video_metadata = metadata
        return self

    def _load_frames(self):
        """
        Load a random subset of frames from the Zarr store into self.frames.
        """
        zarr_store_frames = self._get_zarr_store_frame()
        root_group = zarr.open_group(zarr_store_frames, mode='r')
        data = root_group['data']

        n = self.number_of_frames_to_use
        total_frames = data.shape[0]

        # Sample random frame indices
        random_indices = np.random.choice(total_frames, size=n, replace=False)

        # Load frames using the sampled indices
        random_frames = np.stack([data[i] for i in random_indices])

        self.frame_indices = random_indices
        self.frames = random_frames
        return self

    def _compute_contrast(self):
        """
        Compute Michelson contrast for each frame in self.frames.

        Returns:
            self
        """
        self.contrast_array = [
            (np.max(frame) - np.min(frame)) / (np.max(frame) + np.min(frame))
            for frame in self.frames
        ]
        return self

    def _compute_mtf(self):
        """
        Compute the Modulation Transfer Function (MTF) for each frame using FFT.

        Returns:
            self
        """
        self.modulation_transfer_function = []
        for frame in self.frames:
            f = fft2(frame)
            fshift = fftshift(f)
            magnitude_spectrum = np.abs(fshift)
            mtf = magnitude_spectrum / np.max(magnitude_spectrum)
            self.modulation_transfer_function.append(mtf)
        return self

    def _compute_snr(self):
        """
        Compute the Signal-to-Noise Ratio (SNR) for each frame.

        Returns:
            self
        """
        self.snr_array = [np.mean(frame) / np.std(frame) for frame in self.frames]
        return self

    def _compute_edges(self, method='sobel', sobel_ksize=5, canny_threshold1=100, canny_threshold2=200):
        """
        Compute edges for each frame using either Sobel or Canny edge detection.

        Args:
            method (str): 'sobel' or 'canny'.
            sobel_ksize (int): Kernel size for Sobel operator.
            canny_threshold1 (int): First threshold for Canny.
            canny_threshold2 (int): Second threshold for Canny.

        Returns:
            self
        """
        self.frame_edges = []

        for frame in self.frames:
            img_blur = cv2.GaussianBlur(frame, (3, 3), 0)

            if method == 'sobel':
                edges = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=sobel_ksize)
            elif method == 'canny':
                edges = cv2.Canny(image=img_blur, threshold1=canny_threshold1, threshold2=canny_threshold2)
            else:
                raise ValueError("Invalid method. Use 'sobel' or 'canny'.")

            self.frame_edges.append(edges)

        return self

    def _compute_pixel_distribution(self, bins=256):
        """
        Compute a normalized histogram of pixel values across all frames.

        Args:
            bins (int): Number of bins for the histogram.

        Returns:
            self
        """
        pixel_values = self.frames.ravel()

        hist, bin_edges = np.histogram(pixel_values, bins=bins, range=(0, 255), density=True)

        self.pixel_distribution = hist
        self.pixel_bin_edges = bin_edges

        self.pixel_hist_mean = np.mean(pixel_values)
        self.pixel_hist_median = np.median(pixel_values)
        self.pixel_hist_std = np.std(pixel_values)
        self.pixel_hist_skewness = (
            3 * (self.pixel_hist_mean - self.pixel_hist_median) / self.pixel_hist_std
        )
        return self

    def _detect_blur_laplacian(self):
        """
        Compute the variance of Laplacian for each frame to quantify blur.

        Returns:
            self
        """
        self.blur_values = []
        for frame in self.frames:
            laplacian = cv2.Laplacian(frame, cv2.CV_32F)
            self.blur_values.append(laplacian.var())
        return self

    def _save(self):
        meta_dict = utils.object_to_dict(self)
        session_name = self.video_metadata['session_name']
        with open(Path(RESULTS_PATH, session_name +"_VideoStats.pkl", "wb")) as f:
            pickle.dump(meta_dict, f)
        print('saved object as a dicitonary to json file')
