# Example for a quick test (e.g., notebooks/test_load.py or temporarily in main.py)
import os
import sys
# Add src to Python path to import custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from dicom_utils import load_dicom_image, display_image

INPUT_IMAGE_FOLDER = "/Users/harshit/Documents/M.tech/self/Company/adaptive_iopa_preprocessing/data" # Adjust path if running from notebooks/
# INPUT_IMAGE_FOLDER = "data/" # If running from project root for a script in root

if __name__ == "__main__":
    if not os.path.exists(INPUT_IMAGE_FOLDER) or not os.listdir(INPUT_IMAGE_FOLDER):
        print(f"Error: Input folder '{INPUT_IMAGE_FOLDER}' is empty or does not exist.")
    else:
        print(f"Attempting to load images from: {os.path.abspath(INPUT_IMAGE_FOLDER)}")
        image_files = [f for f in os.listdir(INPUT_IMAGE_FOLDER) if f.lower().endswith(('.dcm', '.rvg'))]
        print(f"Found files: {image_files}")

        if not image_files:
            print("No .dcm or .rvg files found in the data folder.")

        for filename in image_files:
            filepath = os.path.join(INPUT_IMAGE_FOLDER, filename)
            print(f"\n--- Loading: {filename} ---")
            image_array, dcm_data = load_dicom_image(filepath)

            if image_array is not None:
                print(f"Successfully loaded. Image shape: {image_array.shape}, dtype: {image_array.dtype}")
                # display_image(image_array, title=filename) # Uncomment to view each
                if dcm_data:
                    # print("Relevant Metadata:")
                    # for key, value in extract_relevant_metadata(dcm_data).items():
                    #     print(f"  {key}: {value}")
                    pass # Already handled in dicom_utils if needed
            else:
                print(f"Failed to load {filename}.")
        print("\n--- Loading test complete ---")