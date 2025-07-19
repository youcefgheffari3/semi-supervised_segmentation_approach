import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models.unet import UNet
from datasets.weak_dataset import WeakVOCDataset
from utils.voc_utils import load_image_names

# -----------------------
# Configurations
# -----------------------
image_dir = "/content/VOCdevkit/VOC2012/JPEGImages"
annotation_dir = "/content/VOCdevkit/VOC2012/Annotations"
train_list = "/content/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt"
logits_dir = "/content/outputs/ancillary_logits"
os.makedirs(logits_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1  # Weak set processed image-by-image

# -----------------------
# Load Weak Set (Images + BBox Heatmaps)
# -----------------------
weak_images = load_image_names(train_list)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

weak_dataset = WeakVOCDataset(
    image_dir=image_dir,
    annotation_dir=annotation_dir,
    image_list=weak_images,
    transform=transform
)

weak_loader = DataLoader(weak_dataset, batch_size=batch_size, shuffle=False)

# -----------------------
# Load Trained Ancillary Model
# -----------------------
model = UNet(in_channels=4, out_channels=21).to(device)
model.load_state_dict(torch.load("/content/saved_models/ancillary_model.pth"))
model.eval()

# -----------------------
# Generate and Save Logits
# -----------------------
with torch.no_grad():
    for inputs, image_name in weak_loader:
        inputs = inputs.to(device)

        logits = model(inputs)  # Shape: [1, 21, 256, 256]

        # Save logits before softmax
        logits_cpu = logits.squeeze(0).cpu()  # [21, 256, 256]
        save_path = os.path.join(logits_dir, f"{image_name[0]}.pt")
        torch.save(logits_cpu, save_path)

print(f"\n✅ Ancillary logits saved to: {logits_dir}")
