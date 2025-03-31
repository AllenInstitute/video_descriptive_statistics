# Get some stats about the video quality.

📊 **VideoStats**
VideoStats is a Python class for computing various image quality and statistical metrics from a subset of video frames stored in a Zarr file. It includes tools for contrast, signal-to-noise ratio (SNR), blur detection, pixel distribution, edge detection, and frequency-domain analysis (MTF).

📦 **Dependencies**

```
pip install numpy zarr opencv-python scipy
```


🗂 **Input**
 - Zarr directory containing video frames under a data key.

 - metadata attribute stored in the root group (as a JSON string).

 🧠 **Class Overview**
Initialization

```
from pathlib import Path
from video_stats import VideoStats

video_stats = VideoStats(frame_zarr_path=Path("path/to/zarr"), number_of_frames_to_use=100)
```

🛠 **Usage**
1. Load metadata and frames
```
video_stats._load_metadata()._load_frames()
```

2. Compute metrics
```
video_stats._compute_contrast()
video_stats._compute_snr()
video_stats._compute_mtf()
video_stats._compute_pixel_distribution()
video_stats._compute_edges(method='sobel')  # or method='canny'
video_stats._detect_blur_laplacian()
```

💾 **Save results**

Note: _save() assumes a utils.object_to_dict() utility function is available.
```
video_stats._save()
```

📘 **Output Attributes**
- contrast_array: List of Michelson contrast values.

- snr_array: List of SNR values.

- modulation_transfer_function: List of normalized FFT magnitudes.

- frame_edges: List of edge-detected frames.

- pixel_distribution: Histogram of pixel values.

- pixel_hist_mean, pixel_hist_median, pixel_hist_std, pixel_hist_skewness: Summary stats.

- blur_values: Variance of Laplacian (blur metric).

- frame_indices: Indices of randomly selected frames.

🧰 **Notes**
- All computation methods return self, so you can chain them.

- Designed for grayscale image frames.

- To support saving, make sure self.metadata is defined and contains a 'session_name' field.

- utils.object_to_dict() must convert the class object into a dictionary before pickling.
