"""
ORB Feature Matching and ROI Analysis
-----------------------------------------------
This script performs ORB (Oriented FAST and Rotated BRIEF) feature matching between two images.
It supports both RGB and grayscale images. The script loads the images in color (if available)
for display and matching, then converts them to grayscale for ORB detection. Global feature matches 
are computed and drawn using the original color images, and a high-resolution global match image is saved.
Next, the script enters an interactive mode that allows you to select a Region of Interest (ROI)
on the first image. It filters the matches to include only those whose keypoints in Image 1 fall
within the ROI, displays the filtered matches along with the number of matches in that area,
and saves the ROI match image as "ORB_ROI.png".

https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html

Interpretation:
    - The total number of good matches indicates the overall similarity between the images.
      More matches generally imply higher similarity.
    - When using ROI selection, the matches within the ROI focus on a specific area of interest,
      allowing localized analysis.

Usage:
    python ORB.py --image1 /path/to/image1.png --image2 /path/to/image2.png --output /path/to/output/

Note:
    If you encounter "QCoreApplication::exec: The event loop is already running", consider switching 
    the matplotlib backend by adding the following lines at the very top of the script:
        import matplotlib
        matplotlib.use('TkAgg')
"""

import argparse
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#Global variables to store ORB data for ROI selection and output directory
kp1_global = None
kp2_global = None
good_matches_global = None
img1_global = None
img2_global = None
output_dir_global = None

def compute_and_save_all_matches(image1_path, image2_path, output_dir):
    """
    Compute ORB keypoints, descriptors, and matches between two images
    Supports both RGB and grayscale images: images are loaded in color (if available) 
    for display and matching, then converted to grayscale for ORB detection
    The global match image is drawn using the original color images and saved to the output directory

    Args:
        image1_path (str): Path to the first image
        image2_path (str): Path to the second image
        output_dir (str or Path): Directory where the global match image will be saved
    """
    #loading images in color
    img1_orig = cv2.imread(str(image1_path), cv2.IMREAD_COLOR)
    img2_orig = cv2.imread(str(image2_path), cv2.IMREAD_COLOR)

    if img1_orig is None or img2_orig is None:
        print("Error loading images")
        return

    #converting images to grayscale for ORB detection
    img1_gray = cv2.cvtColor(img1_orig, cv2.COLOR_BGR2GRAY) if len(img1_orig.shape) == 3 else img1_orig.copy()
    img2_gray = cv2.cvtColor(img2_orig, cv2.COLOR_BGR2GRAY) if len(img2_orig.shape) == 3 else img2_orig.copy()

    #Initializing ORB detector and compute keypoints and descriptors
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1_gray, None)
    kp2, des2 = orb.detectAndCompute(img2_gray, None)

    #Matching descriptors using BFMatcher with Hamming distance (crossCheck enabled)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    #Sorting matches by distance (lower distance means better match)
    good_matches = sorted(matches, key=lambda x: x.distance)
    total_matches = len(good_matches)
    print(f"Total number of good matches: {total_matches}")

    #Drawing matches using original (color) images for better visualization
    img_matches = cv2.drawMatches(img1_orig, kp1, img2_orig, kp2, good_matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #Converting to RGB for matplotlib display
    img_matches_rgb = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 7))
    plt.imshow(img_matches_rgb)
    plt.text(img1_orig.shape[1]//2 - 50, -10, 'Image 1', fontsize=8, color='red', fontweight='medium')
    plt.text(img1_orig.shape[1] + img2_orig.shape[1]//2 - 50, -10, 'Image 2', fontsize=8, color='green', fontweight='medium')
    plt.title(f'ORB Feature Matching (Total Matches: {total_matches})')
    plt.axis('off')
    output_match_path = Path(output_dir) / "ORB_matches.png"
    plt.savefig(str(output_match_path), dpi=300, bbox_inches='tight')
    plt.close()
    print("Global ORB match image saved as 'ORB_matches.png'")

    #storing data in global variables for use in ROI selection
    global kp1_global, kp2_global, good_matches_global, img1_global, img2_global
    kp1_global = kp1
    kp2_global = kp2
    good_matches_global = good_matches
    img1_global = img1_orig
    img2_global = img2_orig

def select_roi_and_check_matches():
    """
    Interactive ROI selection on the first image. The user clicks and drags on Image 1 to define an ROI
    After releasing the mouse, the ROI selection window closes automatically
    The function then filters ORB matches to only include those whose keypoints in Image 1 fall within the ROI,
    displays the ROI match image along with the count, and saves it as 'ORB_ROI.png' in the output directory
    """
    global kp1_global, kp2_global, good_matches_global, img1_global, img2_global, output_dir_global

    if kp1_global is None:
        print("Feature matches have not been computed yet")
        return

    kp1 = kp1_global
    kp2 = kp2_global
    good_matches = good_matches_global
    img1 = img1_global
    img2 = img2_global

    roi_corners = []
    rect_patch = None
    press = False

    def on_mouse_press(event):
        nonlocal roi_corners, rect_patch, press
        if event.inaxes:
            press = True
            roi_corners = [(event.xdata, event.ydata)]
            if rect_patch is not None:
                rect_patch.remove()
                rect_patch = None
            plt.draw()

    def on_mouse_release(event):
        nonlocal roi_corners, rect_patch, press
        if event.inaxes and press:
            press = False
            roi_corners.append((event.xdata, event.ydata))
            filter_and_draw_matches()
            plt.close(fig)  #automatically close ROI selection window

    def on_mouse_move(event):
        nonlocal rect_patch, press
        if press and event.inaxes:
            x0, y0 = roi_corners[0]
            x1, y1 = event.xdata, event.ydata
            width = x1 - x0
            height = y1 - y0
            if rect_patch is not None:
                rect_patch.remove()
            rect_patch = Rectangle((x0, y0), width, height, linewidth=2, edgecolor='r', facecolor='none')
            event.inaxes.add_patch(rect_patch)
            plt.draw()

    def filter_and_draw_matches():
        #determining ROI boundaries
        x0, y0 = roi_corners[0]
        x1, y1 = roi_corners[1]
        xmin, xmax = sorted([x0, x1])
        ymin, ymax = sorted([y0, y1])

        #Filtering keypoints in Image 1 that lie within the ROI
        idx_in_roi = [idx for idx, kp in enumerate(kp1) if xmin <= kp.pt[0] <= xmax and ymin <= kp.pt[1] <= ymax]
        good_matches_in_roi = [m for m in good_matches if m.queryIdx in idx_in_roi]
        num_matches_in_roi = len(good_matches_in_roi)
        print(f"Number of good matches in ROI: {num_matches_in_roi}")

        #Drawing ROI matches using original images
        img_matches_roi = cv2.drawMatches(img1, kp1, img2, kp2, good_matches_in_roi, None,
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        img_matches_roi_rgb = cv2.cvtColor(img_matches_roi, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(15, 10))
        plt.imshow(img_matches_roi_rgb)
        plt.title(f'ORB Matches within ROI ({num_matches_in_roi} matches)')
        plt.axis('off')

        output_roi_path = output_dir_global / "ORB_ROI.png"
        plt.savefig(str(output_roi_path), dpi=300, bbox_inches='tight')
        print("ROI match image saved as 'ORB_ROI.png'")
        plt.show()

    #interactive ROI selection on Image 1
    fig, ax = plt.subplots(figsize=(10, 8))
    if len(img1.shape) == 3:
        ax.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    else:
        ax.imshow(img1, cmap='gray')
    ax.set_title('Select ROI on Image 1 (click and drag)')
    fig.canvas.mpl_connect('button_press_event', on_mouse_press)
    fig.canvas.mpl_connect('button_release_event', on_mouse_release)
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    plt.show()

def main():
    """
    Main function for ORB Feature Matching and ROI Analysis
    
    Command-line Arguments:
        --image1: Path to the first image
        --image2: Path to the second image
        --output: Path to the output directory for saving results
    
    Workflow:
        1. Compute ORB keypoints, descriptors, and global matches between the two images, and save the global match image
        2. Launch interactive ROI selection on Image 1
        3. Automatically close the ROI selection window after ROI is defined, then display and save the filtered ORB matches within the ROI
    """
    parser = argparse.ArgumentParser(description="ORB Feature Matching and ROI Analysis Script")
    parser.add_argument("--image1", type=str, required=True, help="Path to the first image")
    parser.add_argument("--image2", type=str, required=True, help="Path to the second image")
    parser.add_argument("--output", type=str, required=True, help="Path to the output directory for saving results")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    global output_dir_global
    output_dir_global = output_dir

    #computing and saving global ORB matches between image1 and image2
    compute_and_save_all_matches(args.image1, args.image2, output_dir)

    #launching interactive ROI selection and display matches within ROI
    select_roi_and_check_matches()

if __name__ == "__main__":
    main()