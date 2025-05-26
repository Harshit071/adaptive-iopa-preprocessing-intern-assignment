# src/compare_formats.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt # If you prefer plt.imsave for consistency with main.py

# Adjust paths if this script is in src/ and data/ is at project root
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)

from dicom_utils import load_dicom_image # Assuming dicom_utils.py is in the same src directory

def create_format_comparison(dcm_path, rvg_path, output_dir):
    """
    Loads one DCM and one RVG image, places them side-by-side, and saves the comparison.
    """
    dcm_filename = os.path.basename(dcm_path)
    rvg_filename = os.path.basename(rvg_path)

    img_dcm, _ = load_dicom_image(dcm_path)
    img_rvg, _ = load_dicom_image(rvg_path)

    if img_dcm is None:
        print(f"Failed to load DCM image: {dcm_path}")
        return
    if img_rvg is None:
        print(f"Failed to load RVG image: {rvg_path}")
        return

    # Ensure both are BGR for consistent stacking and text
    if len(img_dcm.shape) == 2:
        img_dcm_bgr = cv2.cvtColor(img_dcm, cv2.COLOR_GRAY2BGR)
    else:
        img_dcm_bgr = img_dcm

    if len(img_rvg.shape) == 2:
        img_rvg_bgr = cv2.cvtColor(img_rvg, cv2.COLOR_GRAY2BGR)
    else:
        img_rvg_bgr = img_rvg
        
    # Add labels
    cv2.putText(img_dcm_bgr, f"DCM: {dcm_filename}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img_rvg_bgr, f"RVG: {rvg_filename}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Resize images to have the same height for cleaner stacking
    h_dcm, w_dcm = img_dcm_bgr.shape[:2]
    h_rvg, w_rvg = img_rvg_bgr.shape[:2]
    
    target_height = max(h_dcm, h_rvg) # Or a fixed height like 800

    if h_dcm != target_height:
        ratio_dcm = target_height / h_dcm
        img_dcm_bgr = cv2.resize(img_dcm_bgr, (int(w_dcm * ratio_dcm), target_height))
    
    if h_rvg != target_height:
        ratio_rvg = target_height / h_rvg
        img_rvg_bgr = cv2.resize(img_rvg_bgr, (int(w_rvg * ratio_rvg), target_height))

    # Stack side by side
    comparison_image = np.hstack((img_dcm_bgr, img_rvg_bgr))

    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"DCM_vs_RVG_comparison_{os.path.splitext(dcm_filename)[0]}_{os.path.splitext(rvg_filename)[0]}.png"
    save_path = os.path.join(output_dir, output_filename)
    
    try:
        # Using plt.imsave to be consistent if main.py uses it (handles BGR->RGB for PNG)
        # cv2.imwrite saves BGR directly. plt.imsave expects RGB or Grayscale.
        # If images are BGR, convert to RGB for plt.imsave
        comparison_image_rgb = cv2.cvtColor(comparison_image, cv2.COLOR_BGR2RGB)
        plt.imsave(save_path, comparison_image_rgb)
        # Or using cv2.imwrite:
        # cv2.imwrite(save_path, comparison_image)
        print(f"Saved format comparison: {save_path}")
    except Exception as e:
        print(f"Error saving format comparison {save_path}: {e}")


if __name__ == "__main__":
    # --- !!! CHOOSE YOUR IMAGES HERE !!! ---
    # Replace with actual filenames from your data/ folder
    dcm_file_to_compare = "IS20250218_193621_8940_10081171.dcm" 
    rvg_file_to_compare = "R9.rvg"

    path_to_data = os.path.join(_PROJECT_ROOT, "data")
    dcm_full_path = os.path.join(path_to_data, dcm_file_to_compare)
    rvg_full_path = os.path.join(path_to_data, rvg_file_to_compare)

    output_comparison_dir = os.path.join(_PROJECT_ROOT, "results", "format_comparisons")

    if not os.path.exists(dcm_full_path):
        print(f"Error: DCM file not found at {dcm_full_path}")
    elif not os.path.exists(rvg_full_path):
        print(f"Error: RVG file not found at {rvg_full_path}")
    else:
        create_format_comparison(dcm_full_path, rvg_full_path, output_comparison_dir)
        # You can call create_format_comparison multiple times for different pairs
        # For example:
        # create_format_comparison(os.path.join(path_to_data, "another.dcm"), 
        #                          os.path.join(path_to_data, "another.rvg"), 
        #                          output_comparison_dir)