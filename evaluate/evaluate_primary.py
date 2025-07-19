import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models.unet import UNet
from datasets.fully_labeled_dataset import FullyLabeledVOCDataset
from utils.voc_utils import load_image_names
from utils.metrics import compute_iou, compute_mean_iou

# -----------------------
# Configurations
# -----------------------
image_dir = "/content/VOCdevkit/VOC2012/JPEGImages"
annotation_dir = "/content/VOCdevkit/VOC2012/Annotations"
mask_dir = "/content/VOCdevkit/VOC2012/SegmentationClass"
val_list = "/content/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"
model_path = "/content/saved_models/primary_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1  # Evaluation done one-by-one (no randomness)

# -----------------------
# Prepare Validation Data
# -----------------------
val_images = load_image_names(val_list)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_dataset = FullyLabeledVOCDataset(
    image_dir=image_dir,
    annotation_dir=annotation_dir,
    mask_dir=mask_dir,
    image_list=val_images,
    transform=transform
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -----------------------
# Load Trained Primary Model
# -----------------------
model = UNet(in_channels=4, out_channels=21).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# -----------------------
# Evaluate mIoU
# -----------------------
all_ious = []

with torch.no_grad():
    for inputs, masks in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)

        preds = outputs.argmax(dim=1).squeeze(0).cpu()
        masks = masks.squeeze(0).cpu()

        ious = compute_iou(preds, masks)
        all_ious.append(ious)

# -----------------------
# Report Results
# -----------------------
per_class_iou, mean_iou = compute_mean_iou(all_ious)

print("\n📊 Per-Class IoU:")
for class_idx, iou in enumerate(per_class_iou):
    print(f"Class {class_idx}: IoU {iou:.4f}")

print(f"\n🔥 Mean IoU (mIoU): {mean_iou:.4f}")
