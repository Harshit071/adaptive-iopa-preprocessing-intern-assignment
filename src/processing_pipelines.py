# src/processing_pipelines.py

import cv2
import numpy as np
from quality_metrics import calculate_noise_wavelet # For re-evaluating noise before sharpening

# --- Static Preprocessing Pipeline ---
# ENSURE THIS FUNCTION IS PRESENT AND CORRECTLY NAMED
def static_preprocess_image(image):
    if image is None: return None
    processed_image = image.copy()
    processed_image = cv2.medianBlur(processed_image, 3) 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    processed_image = clahe.apply(processed_image)
    gaussian_blur = cv2.GaussianBlur(processed_image, (5, 5), 0)
    processed_image = cv2.addWeighted(processed_image, 1.5, gaussian_blur, -0.5, 0)
    processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)
    return processed_image

# --- Helper function for scaling values ---
def scale_value(value, val_min, val_max, out_min, out_max, invert=False):
    """Scales a value from one range to another, with optional inversion."""
    if value is None:
        return (out_min + out_max) / 2 
    
    value = np.clip(value, val_min, val_max)
    
    denominator = (val_max - val_min)
    if denominator < 1e-6: 
        return out_min if not invert else out_max 

    if invert:
        scaled_val = ( (val_max - value) / denominator ) * (out_max - out_min) + out_min
    else:
        scaled_val = ( (value - val_min) / denominator ) * (out_max - out_min) + out_min
    
    return np.clip(scaled_val, min(out_min, out_max), max(out_min, out_max))


# --- Refined Adaptive Preprocessing Pipeline with Your Thresholds ---
def adaptive_preprocess_image(image, quality_metrics, filename_for_logging="Unknown Image"):
    if image is None or quality_metrics is None:
        print(f"  [{filename_for_logging}] Skipping adaptive processing: image or quality_metrics is None.")
        return None
    
    processed_image = image.copy()

    # Your dataset-specific observed values and targets
    brightness_min_observe = 116.35
    brightness_max_observe = 190.02
    brightness_ideal_mean = 154.83
    contrast_std_min_observe = 29.83
    contrast_std_max_observe = 83.57
    contrast_target_std_ideal = 48.60
    sharpness_lap_var_min_observe = 82.74
    sharpness_lap_var_max_observe = 373.52 
    noise_sigma_min_observe = 0.0040 
    noise_sigma_max_observe = 0.0156 

    brightness = quality_metrics.get("brightness")
    contrast_std = quality_metrics.get("contrast_std")
    sharpness_lap_var = quality_metrics.get("sharpness_laplacian_var")
    noise_sigma = quality_metrics.get("noise_wavelet_sigma")

    brightness_str = f"{brightness:.2f}" if brightness is not None else "N/A"
    contrast_std_str = f"{contrast_std:.2f}" if contrast_std is not None else "N/A"
    sharpness_lap_var_str = f"{sharpness_lap_var:.2f}" if sharpness_lap_var is not None else "N/A"
    noise_sigma_str = f"{noise_sigma:.4f}" if noise_sigma is not None else "N/A"

    print(f"  [{filename_for_logging}] Initial Adaptive Metrics: B={brightness_str}, "
          f"C_std={contrast_std_str}, S_lap={sharpness_lap_var_str}, N_sig={noise_sigma_str}")

    # --- 1. Adaptive Denoising ---
    denoising_trigger_threshold = noise_sigma_min_observe + 0.002 
    median_blur_threshold = noise_sigma_min_observe + 0.005                                                            
    if noise_sigma is not None:
        if noise_sigma > denoising_trigger_threshold:
            if noise_sigma < median_blur_threshold:
                median_ksize = 3
                print(f"  [{filename_for_logging}] Applying Median Blur (k={median_ksize}) for slight noise (sigma: {noise_sigma:.4f})")
                processed_image = cv2.medianBlur(processed_image, median_ksize)
            else: 
                nlm_h_min = 2.0  
                nlm_h_max = 12.0 
                adaptive_nlm_h = scale_value(noise_sigma, median_blur_threshold, noise_sigma_max_observe, 
                                             nlm_h_min, nlm_h_max, invert=False)
                print(f"  [{filename_for_logging}] Applying NLM Denoising with h={adaptive_nlm_h:.2f} (noise sigma: {noise_sigma:.4f})")
                processed_image = cv2.fastNlMeansDenoising(processed_image, h=float(adaptive_nlm_h))
        else:
            print(f"  [{filename_for_logging}] Image considered clean (noise sigma: {noise_sigma:.4f}). Skipping denoising.")
    else:
        print(f"  [{filename_for_logging}] Noise metric unavailable. Skipping denoising.")

    # --- 2. Adaptive Contrast Enhancement (CLAHE) ---
    if contrast_std is not None:
        clahe_clip_min = 0.5  
        clahe_clip_default = 1.5 
        clahe_clip_max_low_contrast = 3.5 
        adaptive_clip_limit = clahe_clip_default
        current_contrast_std_str = f"{contrast_std:.2f}"

        if contrast_std < contrast_std_min_observe + 5: 
            adaptive_clip_limit = scale_value(contrast_std, contrast_std_min_observe, contrast_std_min_observe + 10, 
                                              clahe_clip_max_low_contrast, clahe_clip_default, invert=True) 
            print(f"  [{filename_for_logging}] Very Low contrast (std: {current_contrast_std_str}). CLAHE clipLimit={adaptive_clip_limit:.2f}")
        elif contrast_std < contrast_target_std_ideal - 5: 
             adaptive_clip_limit = scale_value(contrast_std, contrast_std_min_observe + 5, contrast_target_std_ideal -5, 
                                              clahe_clip_default + 0.5, clahe_clip_default, invert=True)
             print(f"  [{filename_for_logging}] Moderately Low contrast (std: {current_contrast_std_str}). CLAHE clipLimit={adaptive_clip_limit:.2f}")
        elif contrast_std > contrast_std_max_observe - 10: 
             adaptive_clip_limit = clahe_clip_min 
             print(f"  [{filename_for_logging}] Very High contrast (std: {current_contrast_std_str}). Minimal CLAHE clipLimit={adaptive_clip_limit:.2f}")
        else: 
            print(f"  [{filename_for_logging}] Acceptable/Moderate contrast (std: {current_contrast_std_str}). Default CLAHE clipLimit={adaptive_clip_limit:.2f}")
        
        if adaptive_clip_limit > 0.1: 
            clahe = cv2.createCLAHE(clipLimit=adaptive_clip_limit, tileGridSize=(8, 8))
            processed_image = clahe.apply(processed_image)
    else:
        print(f"  [{filename_for_logging}] Contrast metric unavailable. Skipping CLAHE.")

    # --- 3. Adaptive Brightness Adjustment (Gamma Correction) ---
    if brightness is not None:
        gamma_min_brighten = 0.6 
        gamma_max_darken = 1.8   
        gamma = 1.0 
        current_brightness_str = f"{brightness:.2f}"

        brightness_lower_bound_ok = brightness_ideal_mean - 15 
        brightness_upper_bound_ok = brightness_ideal_mean + 25 

        if brightness < brightness_lower_bound_ok: 
            gamma = scale_value(brightness, brightness_min_observe, brightness_lower_bound_ok, 
                                gamma_min_brighten, 1.0, invert=False) 
            print(f"  [{filename_for_logging}] Image too dark (mean: {current_brightness_str}). Applying Gamma={gamma:.2f}")
        elif brightness > brightness_upper_bound_ok:
            gamma = scale_value(brightness, brightness_upper_bound_ok, brightness_max_observe, 
                                1.0, gamma_max_darken, invert=False)
            print(f"  [{filename_for_logging}] Image too bright (mean: {current_brightness_str}). Applying Gamma={gamma:.2f}")
        else:
            print(f"  [{filename_for_logging}] Brightness (mean: {current_brightness_str}) acceptable. No gamma correction.")

        if abs(gamma - 1.0) > 0.05: 
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            processed_image = cv2.LUT(processed_image, table)
    else:
        print(f"  [{filename_for_logging}] Brightness metric unavailable. Skipping Gamma correction.")

    # --- 4. Adaptive Sharpening (Unsharp Masking) ---
    sharpen_alpha_min = 0.1 
    sharpen_alpha_max = 1.0 
    if sharpness_lap_var is not None:
        adaptive_sharpen_alpha = 0.0
        sharpening_needed_threshold = sharpness_lap_var_max_observe + 100 
        current_sharpness_str = f"{sharpness_lap_var:.2f}" 

        if sharpness_lap_var < sharpening_needed_threshold:
            adaptive_sharpen_alpha = scale_value(sharpness_lap_var, 
                                                 sharpness_lap_var_min_observe, 
                                                 sharpening_needed_threshold,   
                                                 sharpen_alpha_max,             
                                                 sharpen_alpha_min,             
                                                 invert=True) 
            current_noise_sigma_after_denoise = calculate_noise_wavelet(processed_image) 
            if current_noise_sigma_after_denoise is not None and current_noise_sigma_after_denoise > (noise_sigma_min_observe + 0.004): 
                noise_reduction_for_sharpen_min_noise = noise_sigma_min_observe + 0.004
                noise_reduction_for_sharpen_max_noise = noise_sigma_max_observe 
                reduction_factor = scale_value(current_noise_sigma_after_denoise, 
                                               noise_reduction_for_sharpen_min_noise, 
                                               noise_reduction_for_sharpen_max_noise, 
                                               1.0, 0.3, invert=True) 
                adaptive_sharpen_alpha *= reduction_factor
                print(f"  [{filename_for_logging}] Reducing sharpen alpha by factor {reduction_factor:.2f} due to residual noise ({current_noise_sigma_after_denoise:.4f})")
        
        if adaptive_sharpen_alpha > 0.05: 
            print(f"  [{filename_for_logging}] Applying Unsharp Masking with alpha={adaptive_sharpen_alpha:.2f} (Laplacian Var: {current_sharpness_str})")
            gaussian_blur = cv2.GaussianBlur(processed_image, (0,0), sigmaX=1.0) 
            processed_image = cv2.addWeighted(processed_image, 1 + adaptive_sharpen_alpha, 
                                              gaussian_blur, -adaptive_sharpen_alpha, 0)
            processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)
        else:
            print(f"  [{filename_for_logging}] Sharpness (Lap Var: {current_sharpness_str}) sufficient or alpha too low. Skipping sharpening.")
    else:
        print(f"  [{filename_for_logging}] Sharpness metric unavailable. Skipping sharpening.")

    return processed_image