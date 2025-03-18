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

