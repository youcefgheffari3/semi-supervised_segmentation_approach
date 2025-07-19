import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET


class WeakVOCDataset(Dataset):
    """
    Weakly-Labeled VOC Dataset for Semi-Supervised Segmentation.
    - Provides images + bounding box heatmaps.
    - No segmentation masks (used for generating logits with ancillary model).
    """
    def __init__(self, image_dir, annotation_dir, image_list, transform=None):
        """
        Args:
            image_dir (str): Path to JPEGImages directory.
            annotation_dir (str): Path to Annotations directory (VOC XML files).
            image_list (List[str]): List of image names (without extension).
            transform (torchvision.transforms.Compose): Transformations (Resize, Normalize, ToTensor).
        """
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.image_list = image_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]

        # Load image and annotation paths
        image_path = os.path.join(self.image_dir, f"{image_name}.jpg")
        annotation_path = os.path.join(self.annotation_dir, f"{image_name}.xml")

        # Load RGB image
        image = Image.open(image_path).convert("RGB")

        # Parse bounding boxes and create heatmap
        bboxes = self.parse_voc_annotation(annotation_path)
        image_cv = cv2.imread(image_path)
        h, w = image_cv.shape[:2]
        heatmap = self.create_bbox_heatmap((h, w), bboxes)

        # Apply image transformations (resize, normalize)
        if self.transform:
            image = self.transform(image)

        # Resize heatmap to match image
        heatmap = cv2.resize(heatmap, (256, 256), interpolation=cv2.INTER_NEAREST)
        heatmap = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0)

        # Stack channels: RGB (3) + Heatmap (1) => 4 channels
        input_tensor = torch.cat([image, heatmap], dim=0)

        return input_tensor, image_name

    @staticmethod
    def parse_voc_annotation(xml_file):
        """Extract bounding boxes from VOC XML annotation file."""
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

    @staticmethod
    def create_bbox_heatmap(image_size, bboxes):
        """Create a binary heatmap from bounding boxes (1 inside bbox, 0 elsewhere)."""
        heatmap = np.zeros(image_size, dtype=np.float32)
        for box in bboxes:
            xmin, ymin, xmax, ymax = box['bbox']
            heatmap[ymin:ymax, xmin:xmax] = 1.0
        return heatmap
