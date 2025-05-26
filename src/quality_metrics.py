import numpy as np
import cv2
from skimage.restoration import estimate_sigma # Will use PyWavelets

def calculate_brightness(image):
    if image is None: return None
    return np.mean(image)

def calculate_contrast_std(image):
    if image is None: return None
    return np.std(image)

def calculate_sharpness_laplacian_variance(image):
    if image is None: return None
    if image.dtype != np.uint8:
        img_uint8 = np.clip(image if np.max(image) <=1 else image / (np.max(image) if np.max(image) > 0 else 1.0) * 255.0, 0, 255).astype(np.uint8)
    else:
        img_uint8 = image
    return cv2.Laplacian(img_uint8, cv2.CV_64F).var()

def calculate_noise_wavelet(image):
    if image is None: return None
    if image.dtype != np.float32 and image.dtype != np.float64:
        img_float = image.astype(np.float32) / 255.0
    elif np.max(image) > 1.0001: # Allow for small float inaccuracies if already "normalized"
         img_float = image.astype(np.float32) / 255.0 
    else: # Assumed to be in [0,1] or [-1,1]
        img_float = image.astype(np.float32)
    
    img_float = np.clip(img_float, 0.0, 1.0) # Ensure it's strictly in [0,1] for estimate_sigma

    if img_float.ndim > 2: # Should be handled by dicom_utils, but double check
        img_float = img_float[..., 0] 
    try:
        # For grayscale, channel_axis=None or not providing it works for newer skimage
        # average_sigmas is for older versions, multichannel for even older.
        # Let's try to be compatible:
        try:
            sigma = estimate_sigma(img_float, channel_axis=None, average_sigmas=True)
        except TypeError:
            try:
                sigma = estimate_sigma(img_float, channel_axis=None)
            except TypeError:
                sigma = estimate_sigma(img_float, multichannel=False) # Deprecated
        return sigma
    except ImportError: # Specifically for PyWavelets not found
        print("Error in wavelet noise estimation: PyWavelets is not installed. Please ensure it is installed.")
        return np.std(img_float) # Fallback
    except Exception as e:
        print(f"Error in wavelet noise estimation: {e}. Using fallback (std of image).")
        return np.std(img_float) # Fallback


def analyze_image_quality(image):
    if image is None:
        return {"brightness": None, "contrast_std": None, "sharpness_laplacian_var": None, "noise_wavelet_sigma": None}
    return {
        "brightness": calculate_brightness(image),
        "contrast_std": calculate_contrast_std(image),
        "sharpness_laplacian_var": calculate_sharpness_laplacian_variance(image),
        "noise_wavelet_sigma": calculate_noise_wavelet(image),
    }