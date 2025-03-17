"""
SEWAR Metrics Analysis
-----------------------------------
This script compares two sets of images (a reference folder and a distorted folder) using 
various full-reference image quality metrics provided by the 'sewar' library. Each metric 
is saved into a separate CSV file.

https://pypi.org/project/sewar/

USAGE:
    1. Modify the 'ref_folder' and 'dist_folder' paths to point to the reference images and
       distorted images respectively
    2. Modify the 'output_dir' path to specify where the CSV files should be saved
    3. Run the script. The script checks if both folders have the same number of images, 
       then computes the metrics for each pair of images with matching filenames
    4. This script could be used for comparing authentic image and a synthetic image
    5. python SEWAR.py --ref /path/to/reference_folder --dist /path/to/distorted_folder --output /path/to/output_folder

HOW TO INTERPRET SCORES:
    - Higher is better for PSNR, MSSSIM, SSIM, UQI, and VIFP
    - Lower is better for MSE, RMSE, ERGAS, and SAM
"""

import argparse
import cv2
import os
import numpy as np
import pandas as pd
from skimage import metrics
from sewar import full_ref
from tqdm import tqdm

def main():

    parser = argparse.ArgumentParser(description="SEWAR Metrics Analysis Script")
    parser.add_argument("--ref", type=str, required=True,
                        help="Path to the reference (ground-truth) images folder")
    parser.add_argument("--dist", type=str, required=True,
                        help="Path to the distorted/synthetic images folder")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to the output directory where CSV results will be saved")
    args = parser.parse_args()

    ref_folder = args.ref
    dist_folder = args.dist
    output_dir = args.output

    os.makedirs(output_dir, exist_ok=True)

    ref_images = sorted([f for f in os.listdir(ref_folder) if f.endswith('.png')])
    dist_images = sorted([f for f in os.listdir(dist_folder) if f.endswith('.png')])

    if len(ref_images) != len(dist_images):
        print("Folders have different numbers of images. Please ensure matching sets.")
        return


    # METRIC 1: MSE (Mean Squared Error)
    mse_data = []
    print("Calculating MSE for all images...")
    for ref_img_name, dist_img_name in tqdm(zip(ref_images, dist_images), total=len(ref_images), desc="MSE"):
        ref_img_path = os.path.join(ref_folder, ref_img_name)
        dist_img_path = os.path.join(dist_folder, dist_img_name)

        ref_img = cv2.imread(ref_img_path, 1)
        dist_img = cv2.imread(dist_img_path, 1)

        mse_value = metrics.mean_squared_error(ref_img, dist_img)
        mse_data.append([ref_img_name, mse_value])

    pd.DataFrame(mse_data, columns=['Image_Name', 'MSE']).to_csv(
        os.path.join(output_dir, 'mse.csv'), index=False)
    print("MSE calculation completed. Results saved in mse.csv!")


    # METRIC 2: PSNR (Peak Signal-to-Noise Ratio) using skimage
    psnr_ski_data = []
    print("Calculating PSNR (skimage) for all images...")
    for ref_img_name, dist_img_name in tqdm(zip(ref_images, dist_images), total=len(ref_images), desc="PSNR_ski"):
        ref_img_path = os.path.join(ref_folder, ref_img_name)
        dist_img_path = os.path.join(dist_folder, dist_img_name)

        ref_img = cv2.imread(ref_img_path, 1)
        dist_img = cv2.imread(dist_img_path, 1)

        psnr_value = metrics.peak_signal_noise_ratio(ref_img, dist_img, data_range=None)
        if np.isinf(psnr_value) or np.isnan(psnr_value):
            psnr_value = 0
        psnr_ski_data.append([ref_img_name, psnr_value])

    pd.DataFrame(psnr_ski_data, columns=['Image_Name', 'PSNR_ski']).to_csv(
        os.path.join(output_dir, 'psnr_ski.csv'), index=False)
    print("PSNR (skimage) calculation completed. Results saved in psnr_ski.csv!")


    # METRIC 3: ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse)
    ergas_data = []
    print("Calculating ERGAS for all images...")
    for ref_img_name, dist_img_name in tqdm(zip(ref_images, dist_images), total=len(ref_images), desc="ERGAS"):
        ref_img_path = os.path.join(ref_folder, ref_img_name)
        dist_img_path = os.path.join(dist_folder, dist_img_name)

        ref_img = cv2.imread(ref_img_path, 1)
        dist_img = cv2.imread(dist_img_path, 1)

        ergas_value = full_ref.ergas(ref_img, dist_img, r=4)
        if np.any(np.isnan(ergas_value)) or np.any(np.isinf(ergas_value)):
            ergas_value = 0
        ergas_data.append([ref_img_name, ergas_value])

    pd.DataFrame(ergas_data, columns=['Image_Name', 'ERGAS']).to_csv(
        os.path.join(output_dir, 'ergas.csv'), index=False)
    print("ERGAS calculation completed. Results saved in ergas.csv!")


    # METRIC 4: MSSSIM (Multi-Scale Structural Similarity Index)
    msssim_data = []
    print("Calculating MSSSIM for all images...")
    for ref_img_name, dist_img_name in tqdm(zip(ref_images, dist_images), total=len(ref_images), desc="MSSSIM"):
        ref_img_path = os.path.join(ref_folder, ref_img_name)
        dist_img_path = os.path.join(dist_folder, dist_img_name)

        ref_img = cv2.imread(ref_img_path, 1)
        dist_img = cv2.imread(dist_img_path, 1)

        msssim_value = full_ref.msssim(ref_img, dist_img, 
                                       weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
                                       ws=11, K1=0.01, K2=0.03, MAX=None)
        if np.any(np.isnan(msssim_value)) or np.any(np.isinf(msssim_value)):
            msssim_value = 0
        msssim_data.append([ref_img_name, msssim_value])

    pd.DataFrame(msssim_data, columns=['Image_Name', 'MSSSIM']).to_csv(
        os.path.join(output_dir, 'msssim.csv'), index=False)
    print("MSSSIM calculation completed. Results saved in msssim.csv!")


    # METRIC 5: PSNR (using sewar)
    psnr_data = []
    print("Calculating PSNR (sewar) for all images...")
    for ref_img_name, dist_img_name in tqdm(zip(ref_images, dist_images), total=len(ref_images), desc="PSNR_sewar"):
        ref_img_path = os.path.join(ref_folder, ref_img_name)
        dist_img_path = os.path.join(dist_folder, dist_img_name)

        ref_img = cv2.imread(ref_img_path, 1)
        dist_img = cv2.imread(dist_img_path, 1)

        psnr_value = full_ref.psnr(ref_img, dist_img, MAX=None)
        if np.any(np.isnan(psnr_value)) or np.any(np.isinf(psnr_value)):
            psnr_value = 0
        psnr_data.append([ref_img_name, psnr_value])

    pd.DataFrame(psnr_data, columns=['Image_Name', 'PSNR']).to_csv(
        os.path.join(output_dir, 'psnr.csv'), index=False)
    print("PSNR (sewar) calculation completed. Results saved in psnr.csv!")


    # METRIC 6: RMSE (Root Mean Squared Error)
    rmse_data = []
    print("Calculating RMSE for all images...")
    for ref_img_name, dist_img_name in tqdm(zip(ref_images, dist_images), total=len(ref_images), desc="RMSE"):
        ref_img_path = os.path.join(ref_folder, ref_img_name)
        dist_img_path = os.path.join(dist_folder, dist_img_name)

        ref_img = cv2.imread(ref_img_path, 1)
        dist_img = cv2.imread(dist_img_path, 1)

        rmse_value = full_ref.rmse(ref_img, dist_img)
        if np.any(np.isnan(rmse_value)) or np.any(np.isinf(rmse_value)):
            rmse_value = 0
        rmse_data.append([ref_img_name, rmse_value])

    pd.DataFrame(rmse_data, columns=['Image_Name', 'RMSE']).to_csv(
        os.path.join(output_dir, 'rmse.csv'), index=False)
    print("RMSE calculation completed. Results saved in rmse.csv!")


    # METRIC 7: SAM (Spectral Angle Mapper)
    sam_data = []
    print("Calculating SAM for all images...")
    for ref_img_name, dist_img_name in tqdm(zip(ref_images, dist_images), total=len(ref_images), desc="SAM"):
        ref_img_path = os.path.join(ref_folder, ref_img_name)
        dist_img_path = os.path.join(dist_folder, dist_img_name)

        ref_img = cv2.imread(ref_img_path, 1)
        dist_img = cv2.imread(dist_img_path, 1)

        sam_value = full_ref.sam(ref_img, dist_img)
        if np.any(np.isnan(sam_value)) or np.any(np.isinf(sam_value)):
            sam_value = 0
        sam_data.append([ref_img_name, sam_value])

    pd.DataFrame(sam_data, columns=['Image_Name', 'SAM']).to_csv(
        os.path.join(output_dir, 'sam.csv'), index=False)
    print("SAM calculation completed. Results saved in sam.csv!")


    # METRIC 8: SSIM (Structural Similarity Index)
    ssim_data = []
    print("Calculating SSIM for all images...")
    for ref_img_name, dist_img_name in tqdm(zip(ref_images, dist_images), total=len(ref_images), desc="SSIM"):
        ref_img_path = os.path.join(ref_folder, ref_img_name)
        dist_img_path = os.path.join(dist_folder, dist_img_name)

        ref_img = cv2.imread(ref_img_path, 1)
        dist_img = cv2.imread(dist_img_path, 1)

        ssim_value = full_ref.ssim(ref_img, dist_img, ws=11, K1=0.01, K2=0.03, MAX=None, fltr_specs=None, mode='valid')
        if np.any(np.isnan(ssim_value)) or np.any(np.isinf(ssim_value)):
            ssim_value = 0
        ssim_data.append([ref_img_name, ssim_value])

    pd.DataFrame(ssim_data, columns=['Image_Name', 'SSIM']).to_csv(
        os.path.join(output_dir, 'ssim.csv'), index=False)
    print("SSIM calculation completed. Results saved in ssim.csv!")


    # METRIC 9: UQI (Universal Quality Index)
    uqi_data = []
    print("Calculating UQI for all images...")
    for ref_img_name, dist_img_name in tqdm(zip(ref_images, dist_images), total=len(ref_images), desc="UQI"):
        ref_img_path = os.path.join(ref_folder, ref_img_name)
        dist_img_path = os.path.join(dist_folder, dist_img_name)

        ref_img = cv2.imread(ref_img_path, 1)
        dist_img = cv2.imread(dist_img_path, 1)

        uqi_value = full_ref.uqi(ref_img, dist_img, ws=8)
        if np.any(np.isnan(uqi_value)) or np.any(np.isinf(uqi_value)):
            uqi_value = 0
        uqi_data.append([ref_img_name, uqi_value])

    pd.DataFrame(uqi_data, columns=['Image_Name', 'UQI']).to_csv(
        os.path.join(output_dir, 'uqi.csv'), index=False)
    print("UQI calculation completed. Results saved in uqi.csv!")


    # METRIC 10: VIFP (Visual Information Fidelity - Pixel Domain)
    vifp_data = []
    print("Calculating VIFP for all images...")
    for ref_img_name, dist_img_name in tqdm(zip(ref_images, dist_images), total=len(ref_images), desc="VIFP"):
        ref_img_path = os.path.join(ref_folder, ref_img_name)
        dist_img_path = os.path.join(dist_folder, dist_img_name)

        ref_img = cv2.imread(ref_img_path, 1)
        dist_img = cv2.imread(dist_img_path, 1)

        vifp_value = full_ref.vifp(ref_img, dist_img, sigma_nsq=2)
        if np.any(np.isnan(vifp_value)) or np.any(np.isinf(vifp_value)):
            vifp_value = 0
        vifp_data.append([ref_img_name, vifp_value])

    pd.DataFrame(vifp_data, columns=['Image_Name', 'VIFP']).to_csv(
        os.path.join(output_dir, 'vifp.csv'), index=False)
    print("VIFP calculation completed. Results saved in vifp.csv!")

if __name__ == "__main__":
    main()
