import pydicom
import matplotlib.pyplot as plt # Keep for display_image, though main.py uses plt.imsave
import numpy as np
import os

def load_dicom_image(filepath):
    try:
        ds = pydicom.dcmread(filepath)
        image = ds.pixel_array
        if image.dtype != np.uint8:
            image = image.astype(np.float32)
            min_val, max_val = np.min(image), np.max(image)
            if max_val == min_val: # Avoid division by zero for blank images
                image = np.zeros_like(image, dtype=np.uint8)
            else:
                image = (image - min_val) / (max_val - min_val + 1e-6) # Add epsilon
                image = (image * 255).astype(np.uint8)
        
        if image.ndim > 2: # Handle potential multi-channel images (e.g. color overlays)
            if image.shape[-1] == 3 or image.shape[-1] == 4: # Assume last dim is color
                print(f"Warning: Image {filepath} appears to be multi-channel ({image.shape}). Converting to grayscale using Luminosity method.")
                # Standard Luma conversion: R*0.299 + G*0.587 + B*0.114
                # For simplicity if it's 3-channel, average or take first; if 4, ignore alpha
                image = image[..., 0] # Simplistic: take the first channel
                # Or for a more standard grayscale:
                # if image.shape[-1] == 3:
                #     image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # Requires OpenCV import here
                # elif image.shape[-1] == 4:
                #     image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY) # Requires OpenCV import here
            else: # Unknown multi-channel format
                 print(f"Warning: Image {filepath} has unexpected shape {image.shape}. Taking first slice/channel.")
                 image = image[0] if image.ndim == 3 else image # Simplistic fallback


        return image, ds
    except Exception as e:
        print(f"Error loading DICOM file {filepath}: {e}")
        return None, None

def display_image(image, title="Image", cmap='gray'): # This is for your interactive testing
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

def extract_relevant_metadata(dicom_dataset):
    if dicom_dataset is None:
        return {}
    metadata = {}
    tags_to_extract = {
        "PatientID": "PatientID", "StudyInstanceUID": "StudyInstanceUID", 
        "Modality": "Modality", "PhotometricInterpretation": "PhotometricInterpretation",
        "Rows": "Rows", "Columns": "Columns", "PixelSpacing": "PixelSpacing",
        "BitsStored": "BitsStored", "WindowCenter": "WindowCenter", "WindowWidth": "WindowWidth",
    }
    for tag_name, tag_key in tags_to_extract.items():
        metadata[tag_name] = getattr(dicom_dataset, tag_key, None) # Use default None
    return metadata