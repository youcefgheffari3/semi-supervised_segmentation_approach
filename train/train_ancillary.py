import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models.unet import UNet
from datasets.fully_labeled_dataset import FullyLabeledVOCDataset
from utils.voc_utils import load_image_names

# -----------------------
# Configurations
# -----------------------
image_dir = "/content/VOCdevkit/VOC2012/JPEGImages"
annotation_dir = "/content/VOCdevkit/VOC2012/Annotations"
mask_dir = "/content/VOCdevkit/VOC2012/SegmentationClass"
train_list = "/content/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt"
save_dir = "/content/saved_models"
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4
num_epochs = 5
lr = 1e-3

# -----------------------
# Data Preparation
# -----------------------
train_images = load_image_names(train_list)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = FullyLabeledVOCDataset(
    image_dir=image_dir,
    annotation_dir=annotation_dir,
    mask_dir=mask_dir,
    image_list=train_images,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# -----------------------
# Model, Loss, Optimizer
# -----------------------
model = UNet(in_channels=4, out_channels=21).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# -----------------------
# Training Loop
# -----------------------
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, masks in train_loader:
        inputs, masks = inputs.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"[Epoch {epoch+1}/{num_epochs}] Ancillary Loss: {avg_loss:.4f}")

# -----------------------
# Save Trained Model
# -----------------------
torch.save(model.state_dict(), os.path.join(save_dir, "ancillary_model.pth"))
print(f"\n✅ Ancillary model saved at {os.path.join(save_dir, 'ancillary_model.pth')}")
