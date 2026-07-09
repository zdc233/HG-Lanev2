import shutil
import requests
import json
import os
import shutil
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


# Download a generated image
def download_image(image_url, output_path):
    response = requests.get(image_url)
    if response.status_code == 200:
        with open(output_path, 'wb') as file:
            file.write(response.content)
        print(f"Image downloaded successfully: {output_path}")
    else:
        raise Exception(f"Failed to download image: {response.text}")


class Preprocessor:
    def __init__(self, transparency=1.0):
        """
        Set transparency/strength at initialization.
        transparency: 0.0-1.0, lower values mean more transparent (lighter edges)
        """
        self.transparency = transparency

    # Added save_prefix parameter, defaults to None
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

        # ================= Save intermediate-step images =================
        # if save_prefix is not None:
        #    # Ensure the output directory exists
        #    save_dir = os.path.dirname(save_prefix)
        #    if save_dir and not os.path.exists(save_dir):
        #        os.makedirs(save_dir, exist_ok=True)
        #
        #    cv2.imwrite(f"{save_prefix}_1_canny_edges.png", canny_edges)
        #    cv2.imwrite(f"{save_prefix}_2_line_mask.png", line_mask)
        #    cv2.imwrite(f"{save_prefix}_3_dilated_mask.png", dilated_mask)
        #    cv2.imwrite(f"{save_prefix}_4_color_mask.png", color_mask)
        #    cv2.imwrite(f"{save_prefix}_5_color_masked_edges.png", color_masked_edges)
        #    cv2.imwrite(f"{save_prefix}_6_color_masked_edges_canny.png", color_masked_edges_canny)
        #    cv2.imwrite(f"{save_prefix}_7_combine_image.png", combine_image)
        # =========================================================

        # 5. Apply transparency scaling
        combine_image = (combine_image * self.transparency).astype(np.uint8)

        return combine_image

    @staticmethod
    def read_polygons_from_file(file_path):
        """
        Read polygon data from a file (CULane .lines.txt format).
        Each line is a sequence of x y x y ... representing the points along one lane line.
        """
        polygons = []
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) < 4:  # at least two points are required (x1 y1 x2 y2)
                    continue
                # Pair up the coordinates into (x, y)
                polygon = [(float(parts[i]), float(parts[i + 1]))
                           for i in range(0, len(parts), 2)]
                polygons.append(polygon)
        return polygons

    @staticmethod
    def get_color_mask(img):
        """
        Create a color mask to extract white and yellow lane lines.
        :param img: input image (color)
        :return: color mask
        """
        # Convert to the HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define HSV ranges for white and yellow
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 255, 255])
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        # Create white and yellow masks
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Combine the white and yellow masks
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
                os.unlink(file_path)  # delete file or symbolic link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # delete subfolder
        except Exception as e:
            print(f"Failed to delete: {file_path} -> {e}")


# Main function
def main(seed, json_file_path, original_image_filename, canny_image_filename, positive_prompt, negative_prompt,
         output_image_name, strength_ip2p=1):
    # Load the JSON file
    json_data = load_json_file(json_file_path)

    # Update the image paths in the JSON data
    if json_file_path == json_file_path_canny_p2p:
        json_data['12']['inputs']['image'] = original_image_filename  # original image path
        json_data['38']['inputs']['image'] = canny_image_filename  # Canny image path
        json_data['6']['inputs']['text'] = positive_prompt  # positive prompt
        json_data['7']['inputs']['text'] = negative_prompt  # negative prompt
        json_data['3']['inputs']['seed'] = seed  # seed
        json_data['26']['inputs']['strength'] = strength_ip2p
        json_data['37']['inputs']['filename_prefix'] = output_image_name
    elif json_file_path == json_file_path_canny:
        json_data['12']['inputs']['image'] = canny_image_filename  # Canny image path
        json_data['6']['inputs']['text'] = positive_prompt  # positive prompt
        json_data['7']['inputs']['text'] = negative_prompt  # negative prompt
        json_data['3']['inputs']['seed'] = seed  # seed
        json_data['37']['inputs']['filename_prefix'] = output_image_name
    else:
        raise Exception(f"Canny file format not supported: {json_file_path}")
    # Generate the image
    generate_image(json_data, seed, original_image_filename)


# Example usage
if __name__ == "__main__":
    json_file_path_canny_p2p = "v11_canny_p2p.json"  # path to your JSON file
    json_file_path_canny = "v11_canny.json"
    low_threshold = 100
    high_threshold = 200
    transparency_value = 1.0

    # Define the root directories for images and annotation files
    data_root = './data/culane'
    image_root = f"{data_root}/normal"
    canny_root = f"{data_root}/canny"
    annotator_root = f"{data_root}/normal"
    labels = ['snow', 'rain', 'fog', 'night', 'dusk']

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
                # Get the full path of the image
                image_path = os.path.join(root, file)
                # Build the corresponding annotation file path
                annotator_file = file.replace(".jpg", ".lines.txt")
                annotator_path = os.path.join(annotator_root, annotator_file)
                # canny_path
                canny_file = file.replace(".jpg", "_canny.png")
                canny_path = os.path.join(canny_root, canny_file)
                # Check whether the annotation file exists
                if os.path.exists(annotator_path):
                    image_list.append(image_path)
                    canny_list.append(canny_path)
                    annotator_file_list.append(annotator_path)

    image_list = sorted(image_list, key=lambda x: int(re.search(r'normal_(\d+)\.jpg', x).group(1)))
    canny_list = sorted(canny_list, key=lambda x: int(re.search(r'normal_(\d+)_canny\.png', x).group(1)))
    annotator_file_list = sorted(annotator_file_list,
                                 key=lambda x: int(re.search(r'normal_(\d+)\.lines\.txt', x).group(1)))

    # Initialize the Preprocessor and set the transparency parameter (0.0-1.0)
    apply_canny = Preprocessor(transparency=transparency_value)

    for input_image_path, canny_image_path, annotator_file_path in zip(image_list, canny_list, annotator_file_list):
        input_image = cv2.imread(input_image_path)

        if input_image is None:
            print(f"Failed to load image: {input_image_path}")
            continue

        # Extract the image filename (without extension) to use as the save prefix
        base_name = os.path.basename(input_image_path).replace(".jpg", "")
        debug_save_prefix = f"./debug/{base_name}"

        # Pass save_prefix to save intermediate steps; omit it to return only the final result
        detected_map = apply_canny(
            input_image,
            annotator_file_path,
            low_threshold,
            high_threshold,
            lane_width=15,
            save_prefix=debug_save_prefix
        )
        cv2.imwrite(canny_image_path, detected_map)

        # Upload the images
        original_image_filename = upload_image(input_image_path)
        canny_image_filename = upload_image(canny_image_path)

        for label in labels:
            strength_ip2p = 1
            if label == 'night':
                seed = "190435371239247"
                positive_prompt = "change the sky to night, but not change the lane. detailed, 4k"  # positive prompt
                negative_prompt = "Low-quality, blurry, distorted, unrealistic proportions, dull colors, out of focus, messy background, duplicate characters."  # negative prompt
                json_file_path = json_file_path_canny_p2p
            elif label == 'dusk':
                seed = "190435371239249"
                positive_prompt = "change the sky to dusk, random add less sunset, but not change the lane. detailed, 4k"  # positive prompt
                negative_prompt = "Low-quality, blurry, distorted, unrealistic proportions, dull colors, out of focus, messy background, duplicate characters."  # negative prompt
                json_file_path = json_file_path_canny_p2p
            elif label == 'snow':
                seed = "6"
                positive_prompt = "we are driving a car on the highway. daytime. A dusting of snow. "  # positive prompt
                negative_prompt = "unrealistic proportions"  # negative prompt
                strength_ip2p = 1
                json_file_path = json_file_path_canny
            elif label == 'rain':
                seed = "13"
                positive_prompt = "falling rain, daytime"  # positive prompt
                negative_prompt = "Low-quality, distorted, unrealistic proportions, dull colors, out of focus, messy background, duplicate characters, foggy."  # negative prompt
                json_file_path = json_file_path_canny
            elif label == 'fog':
                seed = "30"
                positive_prompt = "mist, daytime"  # positive prompt
                negative_prompt = "Low-quality, blurry, distorted, unrealistic proportions, dull colors, out of focus, messy background, duplicate characters."  # negative prompt
                json_file_path = json_file_path_canny
            else:
                raise ValueError(f"Unknown label: {label}")
            target_path = f"{data_root}/{label}"
            # Check whether the path exists
            if not os.path.exists(target_path):
                os.makedirs(target_path)

            output_image_name = f"{label}_{original_image_filename[7:-4]}.jpg"

            main(seed, json_file_path, original_image_filename, canny_image_filename, positive_prompt,
                 negative_prompt, output_image_name, strength_ip2p)
