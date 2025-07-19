import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET


class WeakSetWithLogitsDataset(Dataset):
    """
    Weak Set Dataset for Semi-Supervised Training.
    - Provides (image + bbox heatmap) as input.
    - Provides saved ancillary logits as pseudo-label supervision.
    """
    def __init__(self, image_dir, annotation_dir, logits_dir, image_list, transform=None):
        """
        Args:
            image_dir (str): Path to VOC JPEGImages.
            annotation_dir (str): Path to VOC Annotations (XML).
            logits_dir (str): Directory where ancillary logits (.pt) are saved.
            image_list (List[str]): List of image names (no extension).
            transform (torchvision.transforms.Compose): Torchvision image transforms.
        """
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.logits_dir = logits_dir
        self.image_list = image_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]

        # Load paths
        image_path = os.path.join(self.image_dir, f"{image_name}.jpg")
        annotation_path = os.path.join(self.annotation_dir, f"{image_name}.xml")
        logits_path = os.path.join(self.logits_dir, f"{image_name}.pt")

        # Load RGB image
        image = Image.open(image_path).convert("RGB")

        # Generate bounding box heatmap
        bboxes = self.parse_voc_annotation(annotation_path)
        image_cv = cv2.imread(image_path)
        h, w = image_cv.shape[:2]
        heatmap = self.create_bbox_heatmap((h, w), bboxes)

        # Apply image transforms (resize, normalize)
        if self.transform:
            image = self.transform(image)

        # Resize heatmap to match transformed image
        heatmap = cv2.resize(heatmap, (256, 256), interpolation=cv2.INTER_NEAREST)
        heatmap = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0)

        # Stack: (RGB channels 3) + heatmap (1) = 4 channels
        input_tensor = torch.cat([image, heatmap], dim=0)

        # Load pre-computed ancillary logits (shape [21, 256, 256])
        ancillary_logits = torch.load(logits_path)

        return input_tensor, ancillary_logits

    @staticmethod
    def parse_voc_annotation(xml_file):
        """Parse VOC XML file to extract bounding box coordinates."""
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
        """Create binary heatmap where bounding boxes are marked as 1.0."""
        heatmap = np.zeros(image_size, dtype=np.float32)
        for box in bboxes:
            xmin, ymin, xmax, ymax = box['bbox']
            heatmap[ymin:ymax, xmin:xmax] = 1.0
        return heatmap
