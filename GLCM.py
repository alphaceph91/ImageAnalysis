"""
GLCM Texture Analysis
-----------------------------------
This script extracts texture features from images using the Gray-Level Co-occurrence 
Matrix (GLCM). It computes several properties including:

    - Contrast: Measures local intensity variations. Higher contrast indicates larger 
      intensity differences between neighboring pixels.
    - Energy: Also known as Angular Second Moment; indicates textural uniformity. A high 
      energy value suggests the texture is very uniform.
    - Homogeneity: Measures the closeness of the distribution of elements in the GLCM to 
      its diagonal. Higher homogeneity implies more uniformity in the image.
    - Correlation: Reflects how correlated a pixel is to its neighbor over the entire image. 
      Values near 1 or -1 indicate a strong linear relationship.
    - Entropy: Measures the randomness in the texture. Higher entropy indicates more 
      complexity or disorder.

These features can be used for texture classification or quality assessment.

https://de.mathworks.com/help/images/texture-analysis-using-the-gray-level-co-occurrence-matrix-glcm.html
https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_glcm.html

References
----------
.. [1] Haralick, RM.; Shanmugam, K.,
       "Textural features for image classification"
       IEEE Transactions on systems, man, and cybernetics 6 (1973): 610-621.
       :DOI:`10.1109/TSMC.1973.4309314`

USAGE:
    For a single image:
        python GLCM.py --image /path/to/image.png --output /path/to/output_directory

    For a directory of images:
        python GLCM.py --dir /path/to/images_directory --output /path/to/output_directory

The script processes all images in the provided directory and saves the results to a CSV file.
"""

import os
import argparse
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage import io

def extract_texture_features(image_path):
    """
    Extract texture features from an image using GLCM.
    
    Steps:
      1. Load the image as grayscale
      2. Scale image pixel values to the 0-255 range
      3. Compute the Gray-Level Co-occurrence Matrix (GLCM) using a distance of 1 and angle 0
      4. Extract properties: contrast, energy, homogeneity, and correlation
      5. Compute the entropy of the GLCM
      
    Args:
        image_path (str): Path to the image file
        
    Returns:
        list: [contrast, energy, homogeneity, correlation, entropy]
    """
    #loading image as grayscale
    image = io.imread(image_path, as_gray=True)
    
    #scaling the image to uint8 format (0-255)
    image = (image * 255).astype(np.uint8)
    
    #computing the Gray-Level Co-occurrence Matrix (GLCM)
    glcm = graycomatrix(image, distances=[1], angles=[0], symmetric=True, normed=True)
    
    #extracting the GLCM properties
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    
    #computing entropy to measure texture randomness (avoid log(0) with a small epsilon)
    epsilon = 1e-10
    entropy = -np.sum(glcm * np.log2(glcm + epsilon))
    
    return [contrast, energy, homogeneity, correlation, entropy]

def extract_features_from_directory(directory):
    """
    Extract GLCM features from all images in a given directory
    
    Args:
        directory (str): Path to the directory containing images
        
    Returns:
        list: A list of tuples. Each tuple contains (filename, contrast, energy, homogeneity, correlation, entropy)
    """
    features_list = []
    for filename in os.listdir(directory):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(directory, filename)
            features = extract_texture_features(image_path)
            features_list.append((filename, *features))
    return features_list

def main():
    
    #ARGUMENT PARSING
    parser = argparse.ArgumentParser(description="GLCM Texture Analysis Script")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single image to process")
    group.add_argument("--dir", type=str, help="Path to a directory of images to process")
    parser.add_argument("--output", type=str, required=True, help="Path to the output directory for saving the CSV file")
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    #SINGLE IMAGE MODE
    if args.image:
        image_path = args.image
        if not os.path.isfile(image_path):
            print(f"Provided image path is not a file: {image_path}")
            return
        
        #Extracting features for the single image
        features = extract_texture_features(image_path)
        #Creating a DataFrame with one row
        df = pd.DataFrame([["{}".format(os.path.basename(image_path)), *features]],
                          columns=['Image_Name', 'Contrast', 'Energy', 'Homogeneity', 'Correlation', 'Entropy'])
        csv_path = os.path.join(output_dir, 'GLCM_features.csv')
        df.to_csv(csv_path, index=False)
        print(f"Features for {os.path.basename(image_path)} saved to {csv_path}")

    #DIRECTORY MODE
    elif args.dir:
        image_dir = args.dir
        if not os.path.isdir(image_dir):
            print(f"Provided directory path is not a directory: {image_dir}")
            return

        features_dataset = extract_features_from_directory(image_dir)
        df = pd.DataFrame(features_dataset, columns=['Image_Name', 'Contrast', 'Energy', 'Homogeneity', 'Correlation', 'Entropy'])
        df = df.sort_values(by='Image_Name')
        csv_path = os.path.join(output_dir, 'GLCM_features.csv')
        df.to_csv(csv_path, index=False)
        print(f"Features for all images in {image_dir} saved to {csv_path}")

if __name__ == "__main__":
    main()
