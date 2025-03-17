import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.feature import local_binary_pattern

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ------------------------------------------------------------------------------------
# LBP PARAMETERS
# ------------------------------------------------------------------------------------
# RADIUS: Defines the radius for the circular neighborhood used by LBP
# N_POINTS: The total number of sampling points in the circular neighborhood = 8 * RADIUS
# LBP_THRESHOLD: The variance threshold used to classify images as poor quality
#                Images with LBP variance below this threshold are labeled as poor quality
# ------------------------------------------------------------------------------------
RADIUS = 3
N_POINTS = 8 * RADIUS
LBP_THRESHOLD = 0.01

def apply_lbp(image: np.ndarray, radius: int = RADIUS, n_points: int = N_POINTS) -> np.ndarray:
    """
    Apply Local Binary Patterns (LBP) on a grayscale image using a specified radius and number of points

    Args:
        image (np.ndarray): Grayscale image as a numpy array
        radius (int): Radius for LBP (default = RADIUS)
        n_points (int): Number of sampling points for LBP (default = 8 * RADIUS)

    Returns:
        np.ndarray: LBP result as a float array (same shape as input image)
    """
    return local_binary_pattern(image, n_points, radius, method="uniform")


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale if needed, compute its LBP, then normalize the result to [0, 255] for better visibility

    Args:
        image (np.ndarray): Input image (RGB or grayscale) as a numpy array

    Returns:
        np.ndarray: An 8-bit LBP image scaled to [0..255]
    """
    #converting the image to grayscale if required
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #computing the LBP result
    lbp_img = apply_lbp(image)

    #normalizing the LBP image to 0-255 for better visualization
    lbp_min, lbp_max = lbp_img.min(), lbp_img.max()
    lbp_norm = (lbp_img - lbp_min) / (lbp_max - lbp_min + 1e-9) * 255
    return lbp_norm.astype(np.uint8)


def analyze_lbp(image: np.ndarray) -> float:
    """
    Compute the variance of the normalized LBP histogram, used to assess image quality

    Args:
        image (np.ndarray): Grayscale image as a numpy array

    Returns:
        float: The variance of the normalized LBP histogram
    """
    #computing the LBP using defined parameters
    lbp = apply_lbp(image)

    #building a histogram with bins from 0 to N_POINTS + 2
    bins = np.arange(0, N_POINTS + 3)
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, N_POINTS + 2))

    #normalizing the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    #returning the variance of the normalized histogram
    variance = np.var(hist)
    return variance


def save_example_lbp_image(example_image_path: Path, output_dir: Path) -> None:
    """
    Process a single example image, save the resulting LBP image, and display it with a white background.

    Args:
        example_image_path (Path): Path to the example image.
        output_dir (Path): Directory to save the processed LBP image.
    """
    #loading the image
    image = cv2.imread(str(example_image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        logging.error(f"Example image not found: {example_image_path}")
        return

    #computing the normalized LBP image for display
    lbp_display = process_image(image)

    plt.figure(figsize=(8, 8), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')
    plt.imshow(lbp_display, cmap='gray')
    plt.title("LBP Example", color='black')
    plt.axis("off")

    #saving the figure with a white facecolor
    output_file = output_dir / f"lbp_{example_image_path.name}.png"
    plt.savefig(str(output_file), bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close()
    logging.info(f"Example LBP image saved to {output_file}")


def plot_and_save_histogram(lbp_variances: list, num_good: int, num_poor: int, output_path: Path) -> None:
    """
    Plot and save a histogram of LBP variances for the entire dataset, with a white background.

    Args:
        lbp_variances (list): List of LBP variance values for each processed image
        num_good (int): Number of good-quality images
        num_poor (int): Number of poor-quality images
        output_path (Path): Path to save the resulting histogram plot
    """
    plt.figure(figsize=(10, 6), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')

    #plotting the histogram of variances
    plt.hist(lbp_variances, bins=50, color="purple", edgecolor="black")
    plt.title("Distribution of LBP Variances in Images", color='black')
    plt.xlabel("LBP Variance", color='black')
    plt.ylabel("Number of Images", color='black')
    plt.grid(True)
    plt.text(0.7, 0.85, f"Good Quality: {num_good}", transform=ax.transAxes, fontsize=12, color="blue")
    plt.text(0.7, 0.80, f"Poor Quality: {num_poor}", transform=ax.transAxes, fontsize=12, color="red")
    plt.tight_layout()
    plt.savefig(str(output_path), bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close()
    logging.info(f"Histogram saved to {output_path}")


def save_dataframe(df: pd.DataFrame, output_csv: Path) -> None:
    """
    Save the pandas DataFrame (containing image names, LBP variances, and quality flags) to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame with columns [Image_Name, LBP_Variance, Is_Poor_Quality].
        output_csv (Path): Path to the output CSV file.
    """
    df.to_csv(str(output_csv), index=False)
    logging.info(f"CSV saved to {output_csv}")


def process_images(image_dir: Path, output_dir: Path, lbp_threshold: float = LBP_THRESHOLD):
    """
    Process all images in a directory: compute LBP variance, save the normalized LBP image, and
    classify images as poor/good quality based on a threshold

    Args:
        image_dir (Path): Directory containing .png/.jpg/.jpeg images
        output_dir (Path): Directory to save the processed LBP images
        lbp_threshold (float): Variance threshold to classify images as poor quality

    Returns:
        (list, list, list): Tuple containing:
            - image_names (list of str): Filenames of processed images
            - lbp_variances (list of float): LBP variance values for each image
            - poor_quality_images (list of str): Filenames classified as poor quality
    """
    image_names = []
    lbp_variances = []
    poor_quality_images = []

    #iterating over multiple images in the directory
    for file_path in image_dir.glob("*"):
        if file_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue

        #loading the image
        image = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            logging.warning(f"Failed to load image: {file_path}")
            continue

        #computing the LBP variance
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        variance = analyze_lbp(gray)

        #storing the results
        image_names.append(file_path.name)
        lbp_variances.append(variance)
        if variance < lbp_threshold:
            poor_quality_images.append(file_path.name)

        #saving the normalized LBP image for display
        lbp_display = process_image(image)
        output_image_path = output_dir / file_path.name
        cv2.imwrite(str(output_image_path), lbp_display)

    return image_names, lbp_variances, poor_quality_images


def process_single_image(image_path: Path, output_dir: Path) -> None:
    """
    Process a single image for LBP analysis: compute LBP variance, print it to the terminal,
    and save the normalized LBP image to the output directory

    Args:
        image_path (Path): Path to the single image
        output_dir (Path): Directory to save the LBP image
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        logging.error(f"Failed to load image: {image_path}")
        return

    #computing LBP variance
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    variance = analyze_lbp(gray)
    logging.info(f"LBP Variance for {image_path.name}: {variance:.4f}")

    #saving the normalized LBP image
    lbp_display = process_image(image)
    output_image_path = output_dir / image_path.name
    cv2.imwrite(str(output_image_path), lbp_display)
    logging.info(f"Processed LBP image saved to {output_image_path}")


def main():
    """
    Main function for LBP detection:
    - If --image is specified, process a single image
    - If --dir is specified, process all images in the directory
    - Requires --output for the output directory
    """
    parser = argparse.ArgumentParser(description="Local Binary Patterns (LBP) Analysis Script")
    parser.add_argument("--image", type=str, help="Path to a single image to process")
    parser.add_argument("--dir", type=str, help="Path to a directory of images to process")
    parser.add_argument("--output", type=str, required=True, help="Path to the output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    #single image mode
    if args.image:
        image_path = Path(args.image)
        if not image_path.is_file():
            logging.error(f"Provided image path is not a file: {image_path}")
            return
        process_single_image(image_path, output_dir)

    #directory with many images mode
    elif args.dir:
        image_dir = Path(args.dir)
        if not image_dir.is_dir():
            logging.error(f"Provided directory path is not a directory: {image_dir}")
            return

        image_names, lbp_variances, poor_quality_images = process_images(image_dir, output_dir)
        num_poor = len(poor_quality_images)
        num_good = len(image_names) - num_poor

        histogram_path = output_dir / "lbp_variance_histogram.png"
        plot_and_save_histogram(lbp_variances, num_good, num_poor, histogram_path)

        df = pd.DataFrame({
            "Image_Name": image_names,
            "LBP_Variance": lbp_variances,
        })
        df["Is_Poor_Quality"] = df["LBP_Variance"] < LBP_THRESHOLD
        df.sort_values(by="Image_Name", inplace=True)

        csv_path = output_dir / "lbp_analysis.csv"
        save_dataframe(df, csv_path)

        logging.info(f"Total images processed: {len(image_names)}")
        logging.info(f"Good quality images: {num_good}")
        logging.info(f"Poor quality images: {num_poor}")

    else:
        logging.error("Please provide either --image or --dir argument")


if __name__ == "__main__":
    main()