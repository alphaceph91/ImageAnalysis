"""
GLCM Compare Script with Multiple Patch Selection
-----------------------------------
This script allows you to compare two images using Gray-Level Co-occurrence Matrix (GLCM)
features. User can interactively select multiple patch centers from each image by clicking. 
The script extracts a patch of a fixed size (default: 21x21) around each selected point, 
computes GLCM properties for each patch, and then displays a scatter plot of the GLCM features
(dissimilarity and correlation). 

GLCM Metrics Explained:
    - Dissimilarity:
        Measures the difference in gray-level values between neighboring pixels
        Higher dissimilarity indicates greater texture variation
    - Correlation:
        Measures the linear dependency between the gray levels of neighboring pixels
        Values close to 1 (or -1) suggest strong correlation; values near 0 indicate weak correlation

Usage:
    For comparing two images interactively (default 5 patches):
        python GLCM_compare.py --image1 /path/to/image1.png --image2 /path/to/image2.png --output /path/to/output/ --n_points 5 --patch_size 21

If directories are provided, the script selects the first image from each directory.
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage import io

#CONSTANTS
DEFAULT_PATCH_SIZE = 21  #default patch size (patch_size x patch_size)
DEFAULT_N_POINTS = 5     #default number of patches to select interactively

#GLCM parameters for feature computation on each patch
GLCM_DISTANCE = 5        #distance between pixel pairs for GLCM
GLCM_ANGLE = 0           #angle (in radians) for GLCM computation

def load_image(image_path):
    """
    Load an image as grayscale

    Args:
        image_path (str or Path): Path to the image

    Returns:
        np.ndarray: Grayscale image
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The specified image path does not exist: {image_path}")
    return io.imread(image_path, as_gray=True)

def get_patch(image, center, patch_size):
    """
    Extract a square patch from the image centered at the specified coordinate

    Args:
        image (np.ndarray): Grayscale image
        center (tuple): (x, y) coordinates for the center of the patch
        patch_size (int): Size of the square patch

    Returns:
        np.ndarray: Extracted patch.
    """
    half = patch_size // 2
    x, y = int(center[0]), int(center[1])
    #ensuring the indices are within the image boundaries
    x_start = max(0, x - half)
    y_start = max(0, y - half)
    x_end = min(image.shape[1], x + half)
    y_end = min(image.shape[0], y + half)
    return image[y_start:y_end, x_start:x_end]

def compute_glcm_features(patch):
    """
    Compute GLCM features (dissimilarity and correlation) for a given image patch

    Args:
        patch (np.ndarray): Grayscale patch

    Returns:
        tuple: (dissimilarity, correlation)
    """
    glcm = graycomatrix(patch, distances=[GLCM_DISTANCE], angles=[GLCM_ANGLE],
                        levels=256, symmetric=True, normed=True)
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    return dissimilarity, correlation

def interactive_patch_selection(image, title="Select Patch Centers", n_points=DEFAULT_N_POINTS):
    """
    Display the image and let the user select multiple patch centers by clicking

    Args:
        image (np.ndarray): Grayscale image
        title (str): Title for the interactive window
        n_points (int): Number of patch centers to select

    Returns:
        list: List of (x, y) coordinates for the selected patch centers
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")
    pts = plt.ginput(n_points)
    plt.close()
    if pts:
        return pts
    else:
        raise ValueError("No points were selected.")

def draw_patch_rectangles(ax, centers, patch_size):
    """
    Draw rectangles on an axis for each selected patch center

    Args:
        ax (matplotlib.axes.Axes): The axes on which to draw
        centers (list): List of (x, y) coordinates
        patch_size (int): Size of the patch
    """
    half = patch_size // 2
    for center in centers:
        rect = plt.Rectangle((center[0] - half, center[1] - half), patch_size, patch_size,
                             edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(rect)

def main():

    parser = argparse.ArgumentParser(description="GLCM Compare Script with Interactive Patch Selection")
    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument("--image1", type=str, help="Path to the first image")
    group1.add_argument("--dir1", type=str, help="Path to a directory of images for the first set")
    group2 = parser.add_mutually_exclusive_group(required=True)
    group2.add_argument("--image2", type=str, help="Path to the second image")
    group2.add_argument("--dir2", type=str, help="Path to a directory of images for the second set")
    parser.add_argument("--output", type=str, required=True, help="Path to the output directory to save the plot")
    parser.add_argument("--patch_size", type=int, default=DEFAULT_PATCH_SIZE, help="Size of the patch to extract (default: 21)")
    parser.add_argument("--n_points", type=int, default=DEFAULT_N_POINTS, help="Number of patch centers to select (default: 5)")
    args = parser.parse_args()

    #IMAGE LOADING LOGIC
    #For each image set, if a single image is provided, use it;
    #otherwise, if a directory is provided, select the first image
    if args.image1:
        image1_path = args.image1
    else:
        images1 = sorted([f for f in os.listdir(args.dir1) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        if not images1:
            raise ValueError("No images found in the first directory.")
        image1_path = os.path.join(args.dir1, images1[0])
    
    if args.image2:
        image2_path = args.image2
    else:
        images2 = sorted([f for f in os.listdir(args.dir2) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        if not images2:
            raise ValueError("No images found in the second directory.")
        image2_path = os.path.join(args.dir2, images2[0])
    
    # Load images as grayscale
    img1 = load_image(image1_path)
    img2 = load_image(image2_path)

    #Interactive patch selector
    print("Select patch centers in Image 1 (click", args.n_points, "points)")
    centers1 = interactive_patch_selection(img1, title="Select Patch Centers on Image 1", n_points=args.n_points)
    print("Selected patch centers for Image 1:", centers1)

    print("Select patch centers in Image 2 (click", args.n_points, "points)")
    centers2 = interactive_patch_selection(img2, title="Select Patch Centers on Image 2", n_points=args.n_points)
    print("Selected patch centers for Image 2:", centers2)

    #PATCH EXTRACTION & GLCM FEATURE COMPUTATION
    #For each selected center, extract the patch and compute its GLCM features
    glcm_features1 = [compute_glcm_features(get_patch(img1, pt, args.patch_size)) for pt in centers1]
    glcm_features2 = [compute_glcm_features(get_patch(img2, pt, args.patch_size)) for pt in centers2]

    #Printing out features for each patch in both images
    for idx, features in enumerate(glcm_features1, 1):
        print(f"Image 1, Patch {idx} - Dissimilarity: {features[0]:.4f}, Correlation: {features[1]:.4f}")
    for idx, features in enumerate(glcm_features2, 1):
        print(f"Image 2, Patch {idx} - Dissimilarity: {features[0]:.4f}, Correlation: {features[1]:.4f}")

    fig = plt.figure(figsize=(15, 6))

    #Subplot 1: Scatter plot for GLCM features of patches
    ax1 = fig.add_subplot(1, 3, 1)
    for features in glcm_features1:
        ax1.plot(features[0], features[1], 'go', markersize=8)
    for features in glcm_features2:
        ax1.plot(features[0], features[1], 'bo', markersize=8)
    ax1.set_xlabel("GLCM Dissimilarity")
    ax1.set_ylabel("GLCM Correlation")
    ax1.set_title("GLCM Feature Comparison")
    ax1.legend(['Image 1 Patches', 'Image 2 Patches'])

    #Subplot 2: Display Image 1 with patch rectangles
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(img1, cmap="gray")
    draw_patch_rectangles(ax2, centers1, args.patch_size)
    ax2.set_title("Image 1 with Selected Patches")
    ax2.axis("off")

    #Subplot 3: Display Image 2 with patch rectangles
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(img2, cmap="gray")
    draw_patch_rectangles(ax3, centers2, args.patch_size)
    ax3.set_title("Image 2 with Selected Patches")
    ax3.axis("off")

    plt.tight_layout()

    #saving the comparison plot to the output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_plot_path = os.path.join(args.output, "glcm_comparison.png")
    plt.savefig(output_plot_path, bbox_inches='tight', pad_inches=0.1)
    plt.show()

    print(f"Comparison plot saved to {output_plot_path}")

if __name__ == "__main__":
    main()