!pip install roboflow ultralytics torch torchvision onnx onnxruntime scikit-learn opencv-python tensorflow nvidia-pyindex nvidia-tensorrt zipfile36

import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from ultralytics import YOLO
from roboflow import Roboflow

### Step 2: Mount Google Drive and Define Paths

from google.colab import drive
drive.mount('/content/drive')

# Define the path to the zip file
zip_file_path = '/content/drive/MyDrive/filtered_tiles_output.zip'  # Update this to the actual zip file path

# Define the directory where the dataset will be extracted
dataset_path = '/content/final_tiles'  # Directory to extract files

# Create the output directory for results
output_path = '/content/output_images'
os.makedirs(output_path, exist_ok=True)

# Unzip the dataset if not already extracted
if not os.path.exists(dataset_path):
    import zipfile
    print(f"Extracting {zip_file_path}...")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_path)
    print(f"Dataset extracted to {dataset_path}.")
else:
    print(f"Dataset already exists at {dataset_path}.")

from roboflow import Roboflow

rf = Roboflow(api_key="IT111k9ZCIpOy2nsvyeI")
project = rf.workspace("jexa").project("melanoma-by-arpita")
version = project.version(3)

# Check if dataset exists locally before downloading
if not os.path.exists("yolov11"):
    dataset = version.download("yolov11")
    print(f"Dataset downloaded and stored at: {dataset.location}")
else:
    print("Dataset already downloaded.")

data_yaml_path = os.path.join(dataset.location, 'data.yaml')
if os.path.exists(data_yaml_path):
    model = YOLO("yolo11m-seg.pt")
    model.train(data=data_yaml_path, epochs=100, imgsz=256, batch=16)
    print(f"Model training completed. Best model saved at: {model.best.weights}")
else:
    print(f"Error: Dataset YAML file not found at {data_yaml_path}.")

def plot_combined_green_mask(image, results):
    combined_green_mask = np.zeros_like(image, dtype=np.uint8)

    for mask in results[0].masks.data:  # Access masks from the results
        mask_array = mask.cpu().numpy().astype(np.uint8)
        combined_green_mask[mask_array == 1] = [0, 255, 0]

    output_image = cv2.addWeighted(image, 1, combined_green_mask, 0.5, 0)
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Number of cells: {len(results[0].boxes)}")
    plt.show()

# Load YOLOv11 model after training
model = YOLO('/content/runs/segment/train/weights/best.pt')

# Define the dataset path correctly
dataset_path = '/content/final_tiles'  # Adjust to your actual dataset path
image_path = f"/content/final_tiles/tile_10240_11008.png"  # Update to an actual image file

# Check if the image file exists
if os.path.exists(image_path):
    # Load and preprocess the sample image
    image = cv2.imread(image_path)

    if image is not None:
        resized_image = cv2.resize(image, (256, 256))

        # Perform YOLOv11 inference and measure time
        start_time = time.time()
        results = model(resized_image, imgsz=(256, 256))
        inference_time_yolo = time.time() - start_time

        # Create a mask image with the same size as the original image
        mask_image = np.zeros_like(resized_image)

        # Iterate through detected masks and color them green
        for mask in results[0].masks.data:  # Access mask data
            mask_cpu = mask.cpu().numpy().astype(np.uint8)  # Ensure mask is on CPU and in the right format
            mask_image[mask_cpu == 1] = [0, 255, 0]  # Set mask region to green

        # Combine the mask with the original image using bitwise OR
        output_image = cv2.addWeighted(resized_image, 0.5, mask_image, 0.5, 0)

        # Display only the masked image
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
        plt.title(f"YOLOv11 Inference Time: {inference_time_yolo:.3f}s, Cells Count: {len(results[0].boxes)}")
        plt.axis('off')
        plt.show()

    else:
        print(f"Error: Could not read image at {image_path}")
else:
    print(f"Error: Image file does not exist at {image_path}")

# Ensure dataset_path exists and contains image files
dataset_path = '/content/final_tiles'
image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(('.png', '.jpg'))]

if not image_files:
    raise ValueError("No image files found in the specified path.")

# Metrics initialization
total_inference_time = 0
total_cell_count = 0

# Perform inference
for image_path in image_files:
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping invalid image: {image_path}")
        continue

    resized_image = cv2.resize(image, (256, 256))
    start_time = time.time()
    results = model(resized_image, imgsz=(256, 256))
    inference_time = time.time() - start_time

    total_inference_time += inference_time
    total_cell_count += len(results[0].boxes)



# Final metrics
num_images = len(image_files)
average_inference_time = total_inference_time / num_images if num_images else 0

print(f"Total Images Processed: {num_images}")
print(f"Total Cells Counted: {total_cell_count}")
print(f"Total Inference Time: {total_inference_time:.3f} seconds")
print(f"Average Inference Time per Image: {average_inference_time:.3f} seconds")
