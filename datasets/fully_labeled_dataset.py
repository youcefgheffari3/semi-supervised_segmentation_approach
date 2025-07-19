import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class FullyLabeledVOCDataset(Dataset):
    """
    VOC Fully-Labeled Dataset.
    - Loads images, segmentation masks, and generates bounding box heatmaps.
    - Used to train the Ancillary Network (images + heatmap -> masks).
    """
    def __init__(self, image_dir, annotation_dir, mask_dir, image_list, transform=None):
        """
        Args:
            image_dir (str): Directory with JPEGImages.
            annotation_dir (str): Directory with Annotations (XML files).
            mask_dir (str): Directory with SegmentationClass masks.
            image_list (List[str]): List of image names (without extension).
            transform (torchvision.transforms.Compose): Image transformations (Resize, Normalize, etc.)
        """
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.mask_dir = mask_dir
        self.image_list = image_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]

        image_path = os.path.join(self.image_dir, f"{image_name}.jpg")
        annotation_path = os.path.join(self.annotation_dir, f"{image_name}.xml")
        mask_path = os.path.join(self.mask_dir, f"{image_name}.png")

        # Load RGB image
        image = Image.open(image_path).convert("RGB")

        # Generate bounding box heatmap from VOC annotation
        bboxes = self.parse_voc_annotation(annotation_path)
        image_cv = cv2.imread(image_path)
        h, w = image_cv.shape[:2]
        heatmap = self.create_bbox_heatmap((h, w), bboxes)

        # Load mask (convert from RGB to class indices if needed)
        mask = np.array(Image.open(mask_path))
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]  # Ensure grayscale

        # Resize image with torchvision
        if self.transform:
            image = self.transform(image)

        # Resize mask and heatmap
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        heatmap = cv2.resize(heatmap, (256, 256), interpolation=cv2.INTER_NEAREST)
        heatmap = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0)

        # Stack channels: (3 RGB + 1 heatmap)
        input_tensor = torch.cat([image, heatmap], dim=0)
        mask_tensor = torch.tensor(mask, dtype=torch.long)

        return input_tensor, mask_tensor

    @staticmethod
    def parse_voc_annotation(xml_file):
        """Parse VOC XML annotations to extract bounding boxes."""
        import xml.etree.ElementTree as ET
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
        """Create a binary heatmap where bounding boxes are marked as 1.0."""
        heatmap = np.zeros(image_size, dtype=np.float32)
        for box in bboxes:
            xmin, ymin, xmax, ymax = box['bbox']
            heatmap[ymin:ymax, xmin:xmax] = 1.0
        return heatmap
