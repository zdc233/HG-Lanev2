import shutil
import requests
import json
import os
import cv2
import re
import numpy as np
from PIL import Image

# ComfyUI server address
COMFYUI_API_URL = "http://localhost:8188"


# Load a JSON file
def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


# Upload an image and return its filename
def upload_image(image_path):
    files = {'image': open(image_path, 'rb')}
    response = requests.post(f"{COMFYUI_API_URL}/upload/image", files=files)
    if response.status_code == 200:
        return response.json()['name']
    else:
        raise Exception(f"Failed to upload image: {response.text}")


# Send a request to ComfyUI to generate an image
def generate_image(json_data, seed, original_image_filename):
    # Build the new JSON object
    data = {
        "client_id": "1",
        "prompt": json_data
    }

    # Send the request
    response = requests.post(f"{COMFYUI_API_URL}/prompt", json=data)
    if response.status_code == 200:
        result = response.json()
        print(f"{original_image_filename} submit! (seed: {seed})")
    else:
        raise Exception(f"Failed to generate image: {response.text}")


class Preprocessor:
    def __init__(self, transparency=1.0):
        """
        Set transparency/strength at initialization.
        transparency: 0.0-1.0, lower values mean more transparent (lighter edges)
        """
        self.transparency = transparency

    def __call__(self, img, annotator_file, low_threshold, high_threshold, lane_width=15, save_prefix=None):
        # Apply Canny edge detection to the image
        canny_edges = cv2.Canny(img, low_threshold, high_threshold)

        # Read lane-line polyline points from the annotation file
        polygons = self.read_polygons_from_file(annotator_file)

        # 1. Create a thin-line image: first draw 1px-wide polylines
        h, w = img.shape[:2]
        line_mask = np.zeros((h, w), dtype=np.uint8)

        for polygon in polygons:
            if len(polygon) < 2:
                continue
            pts = np.array(polygon, dtype=np.int32)
            # Draw the polyline (1px wide)
            cv2.polylines(line_mask, [pts], isClosed=False, color=255, thickness=1, lineType=cv2.LINE_AA)

        # 2. Dilate: expand left and right by lane_width pixels
        kernel_size = 2 * lane_width + 1  # total width = 15 + 1 + 15 = 31
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
        dilated_mask = cv2.dilate(line_mask, kernel, iterations=1)

        # 3. Further extract lane lines using color information
        color_mask = self.get_color_mask(img)
        color_masked_edges = cv2.bitwise_and(dilated_mask, color_mask)
        color_masked_edges_canny = cv2.Canny(color_masked_edges, low_threshold, high_threshold)

        # 4. Merge with the Canny edge map
        combine_image = cv2.bitwise_or(canny_edges, color_masked_edges_canny)

        # 5. Apply transparency scaling
        combine_image = (combine_image * self.transparency).astype(np.uint8)

        return combine_image

    @staticmethod
    def read_polygons_from_file(file_path):
        polygons = []
        with open(file_path, 'r') as file:
            for line in file:
                points = line.strip().split()
                polygon = [(float(points[i]), float(points[i + 1])) for i in range(0, len(points), 2)]
                polygons.append(polygon)
        return polygons

    @staticmethod
    def get_color_mask(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 255, 255])
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        color_mask = cv2.bitwise_or(white_mask, yellow_mask)
        return color_mask


def clear_folder(folder_path):
    if not os.path.isdir(folder_path):
        print(f"Path does not exist or is not a folder: {folder_path}")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete: {file_path} -> {e}")


# Main function
def main(seed, json_file_path, original_image_filename, canny_image_filename,
         positive_prompt, negative_prompt, output_image_name, strength_ip2p=1):
    # Load the JSON file
    json_data = load_json_file(json_file_path)

    # Update the image paths in the JSON data
    if json_file_path == json_file_path_canny_p2p:
        json_data['12']['inputs']['image'] = original_image_filename
        json_data['38']['inputs']['image'] = canny_image_filename
        json_data['6']['inputs']['text'] = positive_prompt
        json_data['7']['inputs']['text'] = negative_prompt
        json_data['3']['inputs']['seed'] = seed
        json_data['26']['inputs']['strength'] = strength_ip2p
        json_data['37']['inputs']['filename_prefix'] = output_image_name
    elif json_file_path == json_file_path_canny:
        json_data['12']['inputs']['image'] = canny_image_filename
        json_data['6']['inputs']['text'] = positive_prompt
        json_data['7']['inputs']['text'] = negative_prompt
        json_data['3']['inputs']['seed'] = seed
        json_data['37']['inputs']['filename_prefix'] = output_image_name
    else:
        raise Exception(f"Canny file format not supported: {json_file_path}")

    # Generate the image
    generate_image(json_data, seed, original_image_filename)


# Example usage
if __name__ == "__main__":
    json_file_path_canny_p2p = "v11_canny_p2p.json"
    json_file_path_canny = "v11_canny.json"
    low_threshold = 100
    high_threshold = 200
    transparency_value = 1.0  # transparency_value

    # Define the root directories for images and annotation files
    data_root = './data/culane'
    image_root = f"{data_root}/normal"
    canny_root = f"{data_root}/canny"
    annotator_root = f"{data_root}/normal"

    # Only keep the snow category for batch generation
    labels = ['dusk']

    # Build the lists of image paths
    image_list = []
    canny_list = []
    annotator_file_list = []

    clear_folder('../input')
    clear_folder('../output')

    # Walk through the image root directory
    for root, dirs, files in os.walk(image_root):
        for file in files:
            if file.endswith(".jpg"):
                image_path = os.path.join(root, file)
                annotator_file = file.replace(".jpg", ".lines.txt")
                annotator_path = os.path.join(annotator_root, annotator_file)
                canny_file = file.replace(".jpg", "_canny.png")
                canny_path = os.path.join(canny_root, canny_file)

                if os.path.exists(annotator_path):
                    image_list.append(image_path)
                    canny_list.append(canny_path)
                    annotator_file_list.append(annotator_path)

    image_list = sorted(image_list, key=lambda x: int(re.search(r'normal_(\d+)\.jpg', x).group(1)))
    canny_list = sorted(canny_list, key=lambda x: int(re.search(r'normal_(\d+)_canny\.png', x).group(1)))
    annotator_file_list = sorted(annotator_file_list,
                                 key=lambda x: int(re.search(r'normal_(\d+)\.lines\.txt', x).group(1)))

    # Initialize the Preprocessor and set the transparency parameter (0.0-1.0)
    # 0.3 = 30% strength (more transparent), 0.5 = 50% strength (semi-transparent), 1.0 = original strength (opaque)
    apply_canny = Preprocessor(transparency=transparency_value)

    # Iterate over all input images
    for input_image_path, canny_image_path, annotator_file_path in zip(image_list, canny_list, annotator_file_list):
        input_image = cv2.imread(input_image_path)

        if input_image is None:
            print(f"Failed to load image: {input_image_path}")
            continue

        detected_map = apply_canny(input_image, annotator_file_path, low_threshold, high_threshold)

        # If the image is in RGBA format, save it as PNG to preserve transparency
        if len(detected_map.shape) == 3 and detected_map.shape[2] == 4:
            cv2.imwrite(canny_image_path, detected_map)
        else:
            cv2.imwrite(canny_image_path, detected_map)

        # Upload the images to ComfyUI
        original_image_filename = upload_image(input_image_path)
        canny_image_filename = upload_image(canny_image_path)

        # Process each label
        for label in labels:
            if label == 'snow':
                # Batch-generate images with seeds 0-100
                base_prompt = "we are driving a car on the highway. daytime. A dusting of snow."
                base_negative = "unrealistic proportions"
                json_file_path = json_file_path_canny
                strength_ip2p = 1

                target_path = f"{data_root}/{label}"
                if not os.path.exists(target_path):
                    os.makedirs(target_path)

                # Iterate over seeds from 0 to 100
                for seed_val in range(0, 101):
                    # Append a seed identifier to the filename to distinguish generated images
                    output_image_name = f"{label}_seed{seed_val}_{original_image_filename[7:-4]}"

                    print(f"Generating {label} image with seed {seed_val} for {original_image_filename}")
                    main(seed_val, json_file_path, original_image_filename, canny_image_filename,
                         base_prompt, base_negative, output_image_name, strength_ip2p)

            elif label == 'rain':
                base_prompt = "falling rain, daytime"
                base_negative = "Low-quality, distorted, unrealistic proportions, dull colors, out of focus, messy background, duplicate characters, foggy."
                json_file_path = json_file_path_canny
                strength_ip2p = 1

                target_path = f"{data_root}/{label}"
                if not os.path.exists(target_path):
                    os.makedirs(target_path)

                # Iterate over seeds from 0 to 100
                for seed_val in range(0, 101):
                    # Append a seed identifier to the filename to distinguish generated images
                    output_image_name = f"{label}_seed{seed_val}_{original_image_filename[7:-4]}"

                    print(f"Generating {label} image with seed {seed_val} for {original_image_filename}")
                    main(seed_val, json_file_path, original_image_filename, canny_image_filename,
                         base_prompt, base_negative, output_image_name, strength_ip2p)

            elif label == 'fog':
                base_prompt = "mist, daytime"
                base_negative = "Low-quality, blurry, distorted, unrealistic proportions, dull colors, out of focus, messy background, duplicate characters."
                json_file_path = json_file_path_canny
                strength_ip2p = 1

                target_path = f"{data_root}/{label}"
                if not os.path.exists(target_path):
                    os.makedirs(target_path)
                # Iterate over seeds from 0 to 100
                for seed_val in range(0, 101):
                    # Append a seed identifier to the filename to distinguish generated images
                    output_image_name = f"{label}_seed{seed_val}_{original_image_filename[7:-4]}"

                    print(f"Generating {label} image with seed {seed_val} for {original_image_filename}")
                    main(seed_val, json_file_path, original_image_filename, canny_image_filename,
                         base_prompt, base_negative, output_image_name, strength_ip2p)


            elif label == 'night':
                base_prompt = "change the sky to night, but not change the lane. detailed, 4k"
                base_negative = "Low-quality, blurry, distorted, unrealistic proportions, dull colors, out of focus, messy background, duplicate characters."
                json_file_path = json_file_path_canny_p2p
                strength_ip2p = 1

                target_path = f"{data_root}/{label}"
                if not os.path.exists(target_path):
                    os.makedirs(target_path)

                for seed_val in range(0, 101):
                    output_image_name = f"{label}_seed{seed_val}_{original_image_filename[7:-4]}"
                    print(f"Generating {label} image with seed {seed_val} for {original_image_filename}")
                    main(seed_val, json_file_path, original_image_filename, canny_image_filename,
                         base_prompt, base_negative, output_image_name, strength_ip2p)

            elif label == 'dusk':
                base_prompt = "change the sky to dusk, but not change the lane. detailed, 4k"
                base_negative = "Low-quality, blurry, distorted, unrealistic proportions, dull colors, out of focus, messy background, duplicate characters."
                json_file_path = json_file_path_canny_p2p
                strength_ip2p = 1

                target_path = f"{data_root}/{label}"
                if not os.path.exists(target_path):
                    os.makedirs(target_path)

                # Iterate over seeds from 0 to 9
                for seed_val in range(0, 10):
                    # Append a seed identifier to the filename to distinguish generated images
                    output_image_name = f"{label}_seed{seed_val}_{original_image_filename[7:-4]}"

                    print(f"Generating {label} image with seed {seed_val} for {original_image_filename}")
                    main(seed_val, json_file_path, original_image_filename, canny_image_filename,
                         base_prompt, base_negative, output_image_name, strength_ip2p)

            else:
                raise ValueError(f"Unknown label: {label}")
