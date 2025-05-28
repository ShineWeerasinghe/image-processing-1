# Step 1: Import libraries
import cv2  # For reading and writing images
import numpy as np  # For numerical operations
import os  # For file and directory operations

# Step 2: Define filter functions

def mean_filter(image, kernel_size=3):
    """
    Apply a mean filter to the image using a kernel of size NxN.
    """
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size**2)
    return cv2.filter2D(image, -1, kernel)

def median_filter(image, kernel_size=3):
    """
    Apply a median filter to the image using a kernel of size NxN.
    """
    return cv2.medianBlur(image, kernel_size)

def midpoint_filter(image, kernel_size=3):
    """
    Apply a midpoint filter to the image using a kernel of size NxN.
    """
    padded_image = cv2.copyMakeBorder(image, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2, cv2.BORDER_WRAP)
    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for c in range(image.shape[2]):  # Process each channel
                region = padded_image[i:i + kernel_size, j:j + kernel_size, c]
                min_val = np.min(region)
                max_val = np.max(region)
                output[i, j, c] = (min_val + max_val) // 2

    return output

# Step 3: Main function to process all JPEG files
def apply_filters_in_directory():
    """
    Read all JPEG images from the current directory, apply filters, and save the output.
    """
    # Get all JPEG files in the current directory
    jpeg_files = [f for f in os.listdir('.') if f.lower().endswith('.jpeg')]

    if not jpeg_files:
        print("No JPEG files found in the current directory.")
        return

    for file in jpeg_files:
        # Read the image
        image = cv2.imread(file, cv2.IMREAD_COLOR)  # Read as color image

        # Apply each filter
        mean_filtered = mean_filter(image)
        median_filtered = median_filter(image)
        midpoint_filtered = midpoint_filter(image)

        # Save the filtered images with appropriate names
        cv2.imwrite(f"{file.split('.')[0]}_mean.jpeg", mean_filtered)
        cv2.imwrite(f"{file.split('.')[0]}_median.jpeg", median_filtered)
        cv2.imwrite(f"{file.split('.')[0]}_midpoint.jpeg", midpoint_filtered)

    print("Filtering complete. Check the directory for output files.")

# Step 4: Run the program
if __name__ == "__main__":
    apply_filters_in_directory()
