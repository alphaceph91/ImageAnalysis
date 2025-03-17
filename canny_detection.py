import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
CANNY_THRESHOLDS = (50, 150)  #(min_threshold, max_threshold)
EDGE_COUNT_THRESHOLD = 100    #Edge count threshold to classify poor quality. Adjust the edge count if required

def apply_canny(image: np.ndarray, thresholds: tuple = CANNY_THRESHOLDS) -> np.ndarray:
    """
    Apply Canny edge detection on a grayscale image
    
    Args:
        image: Grayscale image as a numpy array
        thresholds: Tuple containing (min_threshold, max_threshold)
    
    Returns:
        Binary edge map as a numpy array
    """
    return cv2.Canny(image, thresholds[0], thresholds[1])

def process_image(image: np.ndarray) -> np.ndarray:
    """
    Process a single image for edge detection by converting it to grayscale and applying the Canny detector
    
    Args:
        image: Input image as a numpy array
    
    Returns:
        Binary edge map (grayscale image)
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return apply_canny(image)

def count_edges(image: np.ndarray) -> int:
    """
    Count the number of edge pixels using the Canny edge detector
    
    Args:
        image: Grayscale image as a numpy array
    
    Returns:
        Count of nonzero edge pixels
    """
    edges = apply_canny(image)
    return int(np.sum(edges > 0))

def save_example_edge_image(example_image_path: Path, output_dir: Path) -> None:
    """
    Process an example image and save the resulting edge map as a grayscale image
    
    Args:
        example_image_path: Path to the example image
        output_dir: Directory where the processed image is saved
    """
    image = cv2.imread(str(example_image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        logging.error(f"Example image not found: {example_image_path}")
        return

    processed = process_image(image)
    plt.figure(figsize=(8, 8))
    plt.imshow(processed, cmap="gray")
    plt.title("Canny Edge Detection Example")
    plt.axis("off")
    output_file = output_dir / f"canny_edges_{example_image_path.name}.png"
    plt.savefig(str(output_file))
    plt.close()
    logging.info(f"Example edge image saved to {output_file}")

def plot_and_save_histogram(edge_counts: list, num_good: int, num_poor: int, output_path: Path) -> None:
    """
    Plot and save a histogram of edge counts from processed images
    
    Args:
        edge_counts: List of edge counts
        num_good: Number of good-quality images
        num_poor: Number of poor-quality images
        output_path: File path for saving the histogram plot
    """
    plt.figure(figsize=(10, 6))
    plt.hist(edge_counts, bins=50, color="green", edgecolor="black")
    plt.title("Distribution of Edge Counts in Images")
    plt.xlabel("Edge Count (Number of Edge Pixels)")
    plt.ylabel("Number of Images")
    plt.grid(True)
    plt.text(0.7, 0.85, f"Good Quality: {num_good}", transform=plt.gca().transAxes, fontsize=12, color="blue")
    plt.text(0.7, 0.80, f"Poor Quality: {num_poor}",transform=plt.gca().transAxes, fontsize=12, color="red")
    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()
    logging.info(f"Histogram saved to {output_path}")

def save_dataframe(df: pd.DataFrame, output_csv: Path) -> None:
    """
    Save the results DataFrame as a CSV file
    
    Args:
        df: pandas DataFrame containing the results
        output_csv: Path to the output CSV file
    """
    df.to_csv(str(output_csv), index=False)
    logging.info(f"CSV saved to {output_csv}")

def process_images(image_dir: Path, output_dir: Path, edge_threshold: int = EDGE_COUNT_THRESHOLD):
    """
    Process all images in a directory for edge detection
    
    Args:
        image_dir: Directory containing image files
        output_dir: Directory where processed images will be saved
        edge_threshold: Threshold for classifying images as poor quality based on edge count
        
    Returns:
        Tuple of (image_names, edge_counts, poor_quality_images)
    """
    image_names = []
    edge_counts = []
    poor_quality_images = []

    for file_path in image_dir.glob("*"):
        if file_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        image = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            logging.warning(f"Failed to load image: {file_path}")
            continue

        processed = process_image(image)
        gray_for_count = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        count = count_edges(gray_for_count)

        image_names.append(file_path.name)
        edge_counts.append(count)
        if count < edge_threshold:
            poor_quality_images.append(file_path.name)

        output_image_path = output_dir / file_path.name
        cv2.imwrite(str(output_image_path), processed)

    return image_names, edge_counts, poor_quality_images

def process_single_image(image_path: Path, output_dir: Path) -> None:
    """
    Process a single image for edge detection
    
    This function loads one image (grayscale or RGB), computes its edge map,
    prints the edge count to the terminal, and saves the edge map to the output directory
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save the processed edge map
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        logging.error(f"Failed to load image: {image_path}")
        return
    processed = process_image(image)
    gray_for_count = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    count = count_edges(gray_for_count)
    logging.info(f"Edge count for {image_path.name}: {count}")
    output_image_path = output_dir / image_path.name
    cv2.imwrite(str(output_image_path), processed)
    logging.info(f"Processed image saved to {output_image_path}")

def main():
    parser = argparse.ArgumentParser(description="Canny Edge Detection Script")
    parser.add_argument("--image", type=str, help="Path to a single image to process.")
    parser.add_argument("--dir", type=str, help="Path to a directory of images to process.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output directory.")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        image_path = Path(args.image)
        if not image_path.is_file():
            logging.error(f"Provided image path is not a file: {image_path}")
            return
        process_single_image(image_path, output_dir)
    elif args.dir:
        image_dir = Path(args.dir)
        if not image_dir.is_dir():
            logging.error(f"Provided directory path is not a directory: {image_dir}")
            return
        image_names, edge_counts, poor_quality_images = process_images(image_dir, output_dir)
        num_poor = len(poor_quality_images)
        num_good = len(image_names) - num_poor

        histogram_path = output_dir / "canny_histogram.png"
        plot_and_save_histogram(edge_counts, num_good, num_poor, histogram_path)

        df = pd.DataFrame({
            "Image_Name": image_names,
            "Edge_Count": edge_counts,
        })
        df["Is_Poor_Quality"] = df["Edge_Count"] < EDGE_COUNT_THRESHOLD
        df.sort_values(by="Image_Name", inplace=True)
        csv_path = output_dir / "canny_analysis.csv"
        save_dataframe(df, csv_path)

        logging.info(f"Total images processed: {len(image_names)}")
        logging.info(f"Good quality images: {num_good}")
        logging.info(f"Poor quality images: {num_poor}")
    else:
        logging.error("Please provide either --image or --dir argument.")

if __name__ == "__main__":
    main()
    
# python canny_detection.py --image /path/to/image.jpg --output /path/to/output
# python canny_detection.py --dir /path/to/images --output /path/to/output
