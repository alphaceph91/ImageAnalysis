# Image and Texture Analysis Repository

## Overview
This repository comprises a collection of Python scripts dedicated to **image processing, texture analysis, and feature extraction**. Each script employs various computer vision techniques to analyze textures, detect edges, match features, and assess image quality. 

## Features and Implemented Methods

### 1. Edge Detection and Texture Analysis:
- **Sobel Edge Detection (`sobel_detection.py`)** - Detects image edges using the Sobel operator and classifies images based on edge counts.
- **Canny Edge Detection (`canny_detection.py`)** - Applies the Canny edge detector to extract image edges and evaluate image quality.
- **Local Binary Patterns (`LBP_detection.py`)** - Computes LBP histograms to analyze textures and classify image quality.
- **Gray-Level Co-occurrence Matrix (`GLCM.py`)** - Extracts texture features such as contrast, energy, homogeneity, and correlation.
- **GLCM Patch Comparison (`GLCM_compare.py`)** - Compares selected patches from two images using GLCM features.

### 2. Feature Matching:
- **SIFT Feature Matching (`SIFT.py`)** - Extracts keypoints using Scale-Invariant Feature Transform (SIFT) and matches them between two images.
- **ORB Feature Matching (`ORB.py`)** - Uses Oriented FAST and Rotated BRIEF (ORB) to detect keypoints and compare images.

### 3. Full-Reference Image Quality Assessment:
- **SEWAR Metric Analysis (`SEWAR.py`)** - Computes various full-reference quality metrics (PSNR, SSIM, MSSSIM, RMSE, UQI, etc.) to compare distorted images against reference images.

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
