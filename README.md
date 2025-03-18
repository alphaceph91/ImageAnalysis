# Image and Texture Analysis Repository

## Overview
This repository comprises a collection of Python scripts dedicated to **image processing, texture analysis, and feature extraction**. Each script employs various computer vision techniques to analyze textures, detect edges, match features, and assess image quality. 

## Features and Implemented Methods

### 1. Edge Detection and Texture Analysis:
- **Sobel Edge Detection (`sobel_detection.py`)** - Detects image edges using the Sobel operator and classifies images based on edge counts. [Sobel](https://scikit-image.org/docs/0.25.x/api/skimage.filters.html#skimage.filters.sobel)
  - Use Case: Use Sobel operator when you require a simple and quick method for detecting edges, especially to highlight the orientation of edges in relatively noise-free images. It's less computationally intensive and suitable for applications where real-time processing is essential, and noise levels are low.

- **Canny Edge Detection (`canny_detection.py`)** - Applies the Canny edge detector to extract image edges and evaluate image quality. [Canny](https://scikit-image.org/docs/stable/auto_examples/edges/plot_canny.html)
  - Use Case: Use Canny operator when precise and robust edge detection is required, particularly in images with varying noise levels. Canny provides smoother, thinner, and cleaner edge maps, making it suitable for applications where edge continuity and accuracy are critical.

- **Local Binary Patterns (`LBP_detection.py`)** - Computes LBP histograms to analyze textures and classify image quality. [LBP](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_local_binary_pattern.html)
  - Use Case: Use Sobel operator when you require a simple and quick method for detecting edges, especially to highlight the orientation of edges in relatively noise-free images. It's less computationally intensive and suitable for applications where real-time processing is essential, and noise levels are low.

- **Gray-Level Co-occurrence Matrix (`GLCM.py`)** - Extracts texture features such as contrast, energy, homogeneity, and correlation. [GLCM](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_glcm.html)
  - Use Case: GLCM features are useful for analyzing texture features in images is essential, such as in medical imaging, remote sensing, or quality inspection. GLCM captures spatial relationships between pixel intensities, aiding in texture classification and segmentation tasks.

- **GLCM Patch Comparison (`GLCM_compare.py`)** - Compares selected patches from two images using GLCM features. 

### 2. Feature Matching:
- **SIFT Feature Matching (`SIFT.py`)** - Extracts keypoints using Scale-Invariant Feature Transform (SIFT) and matches them between two images. [SIFT](https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html)
  - Use Case: Use SIFT feature matching when high robustness to scale, rotation, and illumination changes is needed, such as in object recognition and image stitching. However, SIFT is more computationally intensive, which may not be suitable for real-time applications.

- **ORB Feature Matching (`ORB.py`)** - Uses Oriented FAST and Rotated BRIEF (ORB) to detect keypoints and compare images. [ORB](https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html)
  - Use Case: Use ORB when fast computation is essential, and some tolerance to scale and rotation variations is acceptable. ORB offers a good balance between performance and computational efficiency, making it suitable for real-time applications.

### 3. Full-Reference Image Quality Assessment:
- **SEWAR Metric Analysis (`SEWAR.py`)** - Computes various full-reference quality metrics (PSNR, SSIM, MSSSIM, RMSE, UQI, etc.) to compare distorted images against reference images. [SEWAR](https://pypi.org/project/sewar/)
  - Use Case: Use SEWAR metrics for performing full-reference image quality assessments by comparing a distorted image against a reference image. These metrics are particularly useful in scenarios such as image compression, transmission, or enhancement, where quantifying the degradation or improvement in image quality is required.


## Usage
Each script is standalone and can be executed via the command line. Below are some example usages:

### Sobel Edge Detection:
```sh
python sobel_detection.py --dir /path/to/images --output /path/to/output/
```

### SIFT Feature Matching:
```sh
python SIFT.py --image1 /path/to/image1.png --image2 /path/to/image2.png --output /path/to/output/
```

### GLCM Texture Analysis:
```sh
python GLCM.py --dir /path/to/images --output /path/to/output/
```

### SEWAR Metric Analysis:
```sh
python SEWAR.py --ref /path/to/reference_folder --dist /path/to/distorted_folder --output /path/to/output_folder/
```

## Output
Each script generates specific outputs, including:
- **Processed Images:** Edge-detected or feature-matched images.
- **CSV Files:** Computed metrics for image quality assessment.
- **Histograms & Plots:** Graphical representations of feature distributions.

## Requirements
Ensure the following dependencies are installed:
```sh
pip install opencv-python numpy matplotlib pandas skimage sewar tqdm
```
