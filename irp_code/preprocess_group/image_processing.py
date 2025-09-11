"""
image_processing.py
This module provides functions for cropping out regions in the image delineated
by bounding boxes. It also includes functions for other image processing tasks
like rotating and resizing.

Requirements:
    - Python 3.x
    - `Pillow` for image handling: `pip install Pillow`
"""

import os
import cv2
from PIL import Image
import base64
from io import BytesIO
# Optional, not added to project dependencies
from inference_sdk import InferenceHTTPClient


# Local object detection inference client for strawberry images
client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="zBr3zXOUwMQ5tpDhcYxp"
)


def rotate_image_to_horizontal(image_path, save_path=None):
    """
    Rotates the image 90 degrees anticlockwise to correct its orientation.

    Parameters:
    - image_path: str, path to the input image.
    - save_path: str or None, if provided, the rotated image
    will be saved to this path.

    Returns:
    - rotated_image: np.ndarray, the rotated image.
    """
    # Load image
    image = cv2.imread(image_path)

    # Rotate 90 degrees anticlockwise
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Optionally save the result
    if save_path:
        cv2.imwrite(save_path, rotated_image)

    return rotated_image


def rotate_images_in_folder(input_folder, output_folder):
    """
    Rotates all images in the input folder using rotate_image_to_horizontal
    and saves them to the output folder.
    Parameters:
    - input_folder: str, path to the folder containing images to be rotated.
    - output_folder: str, path to the folder for storing rotated images.
    """
    os.makedirs(output_folder, exist_ok=True)
    image_files = [
        f
        for f in os.listdir(input_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    for img_file in image_files:
        input_path = os.path.join(input_folder, img_file)
        output_path = os.path.join(output_folder, img_file)
        rotate_image_to_horizontal(input_path, save_path=output_path)
        print(f"Rotated and saved: {output_path}")


def crop_patches_from_bboxes(image_path, label_path, out_dir, patch_size=128):
    """
    Crop patches from an image based on YOLO format bounding boxes.

    Parameters:
    - image_path (str): Path to the input image.
    - label_path (str): Path to the YOLO format label file.
    - out_dir (str): Directory to save the cropped patches.
    - patch_size (int): Size of the cropped patches (default is 128x128).
    NB: Tbis function was created with the help of GitHub Copilot.
    Thoroughly edited and tested by me.
    All comments, and descriptions are of my own writing.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    with open(label_path, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        _, x_center, y_center, bw, bh = map(float, line.strip().split())
        x1 = int((x_center - bw / 2) * w)
        y1 = int((y_center - bh / 2) * h)
        x2 = int((x_center + bw / 2) * w)
        y2 = int((y_center + bh / 2) * h)
        crop = image.crop((x1, y1, x2, y2)).resize((patch_size, patch_size))
        crop.save(os.path.join(out_dir, f"patch_{i}.jpg"))


def process_all_images(image_dir, label_dir, output_dir, patch_size):
    """
    Process all images in the specified directory, cropping patches based on
    bounding boxes defined in label files. Saves cropped patches
    and full scenes in their respective directories, as specified.

    Parameters:
    - image_dir (str): Directory containing input images.
    - label_dir (str): Directory containing label files in YOLO format.
    - output_dir (str): Directory to save cropped patches.
    - patch_size (int): Size of the cropped patches (default is 128x128).
    - RESIZE_FULL (tuple): Size to resize full images (width, height).
    - SCENE_DIR (str): Directory to save full images.
    """
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files
    image_files = [
        f for f in os.listdir(image_dir) if f.endswith((
            ".jpg", ".jpeg", ".png"))
    ]

    print(f"Found {len(image_files)} images to process")

    for img_file in image_files:
        # Extract the ID from the image filename
        img_id = img_file.split(".")[0]

        # Construct the paths
        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, f"{img_id}.txt")

        # Create a subfolder for this image's patches
        img_output_dir = os.path.join(output_dir, img_id)
        os.makedirs(img_output_dir, exist_ok=True)

        # Check if label file exists
        if os.path.exists(label_path):
            print(f"Processing {img_file}...")
            crop_patches_from_bboxes(img_path,
                                     label_path, img_output_dir, patch_size)
            print(
                f"Saved patches for {img_file} in {
                    img_output_dir}"
            )

        else:
            print(f"Warning: No label file found for {img_file}")

    print("Finished processing all images!")


def json_to_yolo(json_data, output_txt_path):
    """
    Converts JSON prediction data to YOLO format and saves it to a text file.
    Parameters:
    - json_data (list): List containing JSON prediction data.
    - output_txt_path (str): Path to save the YOLO format text file.
    """
    if not json_data or not json_data[0]["predictions"].get("predictions"):
        print("No JSON data or object predictions found in the JSON data.")
        return
    yolo_lines = []
    image_width = json_data[0]["predictions"]["image"]["width"]
    image_height = json_data[0]["predictions"]["image"]["height"]

    for obj in json_data[0]["predictions"]["predictions"]:
        class_id = obj["class_id"]
        x_center = obj["x"] / image_width
        y_center = obj["y"] / image_height
        width = obj["width"] / image_width
        height = obj["height"] / image_height
        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} \
            {width:.6f} {height:.6f}"
        yolo_lines.append(yolo_line)

    with open(output_txt_path, "w") as f:
        print(f"Writing YOLO format to {output_txt_path}")
        f.write("\n".join(yolo_lines))


def delete_matching_images(folder_a, folder_b):
    """
    Deletes .jpg files from folder_a if a file with the same name
    exists in folder_b.
    """
    files_a = set(f for f in os.listdir(folder_a) if
                  f.lower().endswith('.jpg'))
    files_b = set(f for f in os.listdir(folder_b) if
                  f.lower().endswith('.jpg'))
    matching_files = files_a & files_b

    for filename in matching_files:
        file_path = os.path.join(folder_a, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")


# Find bounding boxes in all the images in the folder
def process_images_in_folder(input_folder, output_folder,
                             label_folder, inference_client=client):
    """
    Processes all images in the input folder by
    detecting strawberries and drawing bounding boxes
    using the inference client and saves the results to the output folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if
                   f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_file in image_files:
        input_path = os.path.join(input_folder, img_file)
        print(f"Processing image: {input_path}")
        # Run the workflow on the current image
        result = inference_client.run_workflow(
            workspace_name="strawberry-detector-yqk4q",
            workflow_id="detect-count-and-visualize-3",
            images={"image": input_path},
            use_cache=True
        )
        text_file_path = os.path.join(label_folder,
                                      f"{os.path.splitext(img_file)[0]}.txt")
        print(f"Writing results to: {text_file_path}")
        json_to_yolo(result, text_file_path)
        output_image_b64 = result[0].get('output_image', None)
        if output_image_b64:
            image_bytes = base64.b64decode(output_image_b64)
            image_pil = Image.open(BytesIO(image_bytes))
            output_path = os.path.join(output_folder, img_file)
            image_pil.save(output_path)
            print(f"Processed and saved: {output_path}")
        else:
            print(f"No output image for {img_file}")
