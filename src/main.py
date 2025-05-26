import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np

# Get the absolute path of the directory where main.py is located (src/)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the project root (one level up from src/)
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)

# Define paths relative to the project root, then make them absolute
INPUT_IMAGE_FOLDER_REL = "data/"
OUTPUT_FOLDER_REL = "results/"

INPUT_IMAGE_FOLDER_ABS = os.path.join(_PROJECT_ROOT, INPUT_IMAGE_FOLDER_REL)
OUTPUT_FOLDER_ABS = os.path.join(_PROJECT_ROOT, OUTPUT_FOLDER_REL)


# Import your custom modules (now that _THIS_DIR is defined, can ensure src is in path if needed)
# If main.py is in src, direct imports work when running `python src/main.py` from project root.
# For robustness if script is moved or called differently:
import sys
if _THIS_DIR not in sys.path: # Add src directory to Python path if not already there
    sys.path.append(_THIS_DIR)

from dicom_utils import load_dicom_image, extract_relevant_metadata
from quality_metrics import analyze_image_quality
from processing_pipelines import static_preprocess_image, adaptive_preprocess_image


def ensure_output_directories_exist(base_output_folder_abs):
    subfolders = ["original_visuals", "static_processed_visuals", "adaptive_processed_visuals"]
    
    # Create base output folder first if it doesn't exist
    if not os.path.exists(base_output_folder_abs):
        os.makedirs(base_output_folder_abs)
        print(f"Created base output folder: {base_output_folder_abs}")

    for subfolder in subfolders:
        path = os.path.join(base_output_folder_abs, subfolder)
        os.makedirs(path, exist_ok=True)
    print(f"Ensured output subdirectories exist in: {base_output_folder_abs}")


def save_image_visual(image_array, full_save_path, cmap='gray'):
    if image_array is None:
        print(f"  Skipping save for {os.path.basename(full_save_path)} (image is None)")
        return
    try:
        if image_array.dtype != np.uint8:
            print(f"Warning: Image {os.path.basename(full_save_path)} is not uint8 ({image_array.dtype}). Clipping and converting.")
            img_to_save = np.clip(image_array, 0, 255).astype(np.uint8)
        else:
            img_to_save = image_array
        plt.imsave(full_save_path, img_to_save, cmap=cmap)
    except Exception as e:
        print(f"Error saving image {os.path.basename(full_save_path)}: {e}")


def process_dataset(input_folder_abs, output_folder_abs):
    ensure_output_directories_exist(output_folder_abs)

    image_files = [f for f in os.listdir(input_folder_abs) if f.lower().endswith(('.dcm', '.rvg'))]
    if not image_files:
        print(f"No .dcm or .rvg files found in '{input_folder_abs}'. Please check the directory.")
        return

    print(f"Found {len(image_files)} images to process in '{input_folder_abs}'.")
    all_metrics_data = []

    original_visuals_path = os.path.join(output_folder_abs, "original_visuals")
    static_visuals_path = os.path.join(output_folder_abs, "static_processed_visuals")
    adaptive_visuals_path = os.path.join(output_folder_abs, "adaptive_processed_visuals")

    for filename in tqdm(image_files, desc="Processing Images"):
        filepath = os.path.join(input_folder_abs, filename)
        base_filename = os.path.splitext(filename)[0]

        print(f"\n--- Processing: {filename} ---")

        original_image, dcm_data = load_dicom_image(filepath)
        if original_image is None:
            print(f"  Failed to load {filename}. Skipping this file.")
            all_metrics_data.append({"filename": filename, "status": "load_error"})
            continue
        
        print(f"  Loaded {filename} successfully. Shape: {original_image.shape}, dtype: {original_image.dtype}")
        save_image_visual(original_image, os.path.join(original_visuals_path, f"{base_filename}_original.png"))

        metrics_original = analyze_image_quality(original_image.copy())
        metrics_original_record = {"filename": filename, "processing_step": "original", **metrics_original}
        if dcm_data: metrics_original_record.update(extract_relevant_metadata(dcm_data))
        all_metrics_data.append(metrics_original_record)
        print(f"  Original Metrics: { {k: (f'{v:.3f}' if isinstance(v, float) else v) for k, v in metrics_original.items()} }")

        print("  Applying static preprocessing...")
        static_processed_img = static_preprocess_image(original_image.copy())
        if static_processed_img is not None:
            save_image_visual(static_processed_img, os.path.join(static_visuals_path, f"{base_filename}_static.png"))
            metrics_static = analyze_image_quality(static_processed_img.copy())
            all_metrics_data.append({"filename": filename, "processing_step": "static", **metrics_static})
            print(f"  Static Metrics: { {k: (f'{v:.3f}' if isinstance(v, float) else v) for k, v in metrics_static.items()} }")
        else:
            print("  Static preprocessing failed.")
            all_metrics_data.append({"filename": filename, "processing_step": "static", "status": "processing_error"})

        print("  Applying adaptive preprocessing...")
        adaptive_processed_img = adaptive_preprocess_image(original_image.copy(), metrics_original, filename_for_logging=filename)
        if adaptive_processed_img is not None:
            save_image_visual(adaptive_processed_img, os.path.join(adaptive_visuals_path, f"{base_filename}_adaptive.png"))
            metrics_adaptive = analyze_image_quality(adaptive_processed_img.copy())
            all_metrics_data.append({"filename": filename, "processing_step": "adaptive", **metrics_adaptive})
            print(f"  Adaptive Metrics: { {k: (f'{v:.3f}' if isinstance(v, float) else v) for k, v in metrics_adaptive.items()} }")
        else:
            print("  Adaptive preprocessing failed.")
            all_metrics_data.append({"filename": filename, "processing_step": "adaptive", "status": "processing_error"})

    if not all_metrics_data:
        print("No data was processed to generate a metrics report.")
        return

    metrics_df = pd.DataFrame(all_metrics_data)
    report_path = os.path.join(output_folder_abs, "image_quality_metrics_report.csv")
    try:
        metrics_df.to_csv(report_path, index=False, float_format='%.4f')
        print(f"\nMetrics report saved to: {report_path}")
    except Exception as e:
        print(f"Error saving metrics report to {report_path}: {e}")

    print("\nGenerating metric distribution plots...")
    metrics_to_plot = ["brightness", "contrast_std", "sharpness_laplacian_var", "noise_wavelet_sigma"]
    plot_order = ['original', 'static', 'adaptive']

    for metric_name in metrics_to_plot:
        if metric_name not in metrics_df.columns:
            print(f"  Metric '{metric_name}' not found in report, skipping plot.")
            continue
        
        plot_df = metrics_df.dropna(subset=[metric_name])
        if plot_df.empty or len(plot_df['processing_step'].unique()) < 2 : # Need at least 2 steps for meaningful boxplot comparison
            print(f"  Not enough valid data points or distinct processing steps for '{metric_name}' plot after dropping NaNs. Skipping.")
            continue
            
        plt.figure(figsize=(10, 6))
        # Filter plot_df for only the steps present in plot_order and in the data
        valid_steps_for_plot = [step for step in plot_order if step in plot_df['processing_step'].unique()]
        sns.boxplot(data=plot_df[plot_df['processing_step'].isin(valid_steps_for_plot)], 
                    x='processing_step', y=metric_name, order=valid_steps_for_plot)
        plt.title(f'{metric_name.replace("_", " ").title()} Distribution by Processing Step')
        plot_save_path = os.path.join(output_folder_abs, f"{metric_name}_distribution.png")
        try:
            plt.savefig(plot_save_path)
            plt.close() 
            print(f"  Saved plot: {plot_save_path}")
        except Exception as e:
            print(f"Error saving plot {plot_save_path}: {e}")
    
    print("\n--- Processing complete. Check the output folder. ---")

if __name__ == "__main__":
    print(f"Project Root: {_PROJECT_ROOT}")
    print(f"Input Folder (Absolute): {INPUT_IMAGE_FOLDER_ABS}")
    print(f"Output Folder (Absolute): {OUTPUT_FOLDER_ABS}")

    if not os.path.exists(INPUT_IMAGE_FOLDER_ABS):
        print(f"Error: Input folder '{INPUT_IMAGE_FOLDER_ABS}' does not exist.")
    elif not os.listdir(INPUT_IMAGE_FOLDER_ABS):
        print(f"Error: Input folder '{INPUT_IMAGE_FOLDER_ABS}' is empty.")
    else:
        process_dataset(INPUT_IMAGE_FOLDER_ABS, OUTPUT_FOLDER_ABS)
        # ... (README guidance print statements) ...