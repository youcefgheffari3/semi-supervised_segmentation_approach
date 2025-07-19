import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2


def load_image_names(txt_file):
    """
    Load VOC image identifiers from a split .txt file (train.txt, val.txt).

    Args:
        txt_file (str): Path to the VOC split text file.

    Returns:
        List[str]: List of image names (without extension).
    """
    with open(txt_file, 'r') as f:
        image_names = f.read().splitlines()
    return image_names


def parse_voc_annotation(xml_file):
    """
    Parse VOC XML annotation to extract bounding box coordinates.

    Args:
        xml_file (str): Path to VOC XML annotation file.

    Returns:
        List[Dict[str, List[int]]]: List of bounding boxes, each in [xmin, ymin, xmax, ymax].
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bboxes = []

    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        bboxes.append({'bbox': [xmin, ymin, xmax, ymax]})

    return bboxes


def create_bbox_heatmap(image_size, bboxes):
    """
    Generate a binary heatmap from bounding box annotations.

    Args:
        image_size (Tuple[int, int]): Image dimensions (height, width).
        bboxes (List[Dict]): Bounding box coordinates.

    Returns:
        np.ndarray: Binary heatmap of shape (height, width).
    """
    heatmap = np.zeros(image_size, dtype=np.float32)
    for box in bboxes:
        xmin, ymin, xmax, ymax = box['bbox']
        heatmap[ymin:ymax, xmin:xmax] = 1.0
    return heatmap


def visualize_heatmap_on_image(image_path, heatmap):
    """
    Optional: Visualize a heatmap overlay on top of the original image.
    Useful for debugging or presentation figures.

    Args:
        image_path (str): Path to the input image (BGR as read by cv2).
        heatmap (np.ndarray): Binary heatmap.

    Returns:
        np.ndarray: Combined visualization (BGR image with heatmap overlay).
    """
    image = cv2.imread(image_path)
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.addWeighted(image, 0.7, heatmap_color, 0.3, 0)
    return heatmap_color
