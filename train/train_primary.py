import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models.unet import UNet
from datasets.fully_labeled_dataset import FullyLabeledVOCDataset
from datasets.weak_logits_dataset import WeakSetWithLogitsDataset
from utils.voc_utils import load_image_names

# -----------------------
# Configurations
# -----------------------
image_dir = "/content/VOCdevkit/VOC2012/JPEGImages"
annotation_dir = "/content/VOCdevkit/VOC2012/Annotations"
mask_dir = "/content/VOCdevkit/VOC2012/SegmentationClass"
train_list = "/content/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt"
logits_dir = "/content/outputs/ancillary_logits"
save_dir = "/content/saved_models"
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4
num_epochs = 5
lr = 1e-3
alpha_start = 30.0
alpha_end = 0.5

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

# Fully labeled dataset loader
fully_labeled_dataset = FullyLabeledVOCDataset(
    image_dir=image_dir,
    annotation_dir=annotation_dir,
    mask_dir=mask_dir,
    image_list=train_images,
    transform=transform
)
fully_labeled_loader = DataLoader(fully_labeled_dataset, batch_size=batch_size, shuffle=True)

# Weak dataset loader (with ancillary logits)
weak_dataset = WeakSetWithLogitsDataset(
    image_dir=image_dir,
    annotation_dir=annotation_dir,
    logits_dir=logits_dir,
    image_list=train_images,
    transform=transform
)
weak_loader = DataLoader(weak_dataset, batch_size=1, shuffle=True)

# -----------------------
# Model, Loss, Optimizer
# -----------------------
primary_model = UNet(in_channels=4, out_channels=21).to(device)
criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
optimizer = torch.optim.Adam(primary_model.parameters(), lr=lr)

# -----------------------
# Training Loop
# -----------------------
for epoch in range(num_epochs):
    alpha = alpha_start - (alpha_start - alpha_end) * (epoch / (num_epochs - 1))  # Linear schedule
    primary_model.train()
    running_loss = 0.0

    # 1️⃣ Train on fully-labeled data (cross-entropy)
    for inputs, masks in fully_labeled_loader:
        inputs, masks = inputs.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = primary_model(inputs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 2️⃣ Train on weak data (self-correction with logits blending)
    for inputs, ancillary_logits in weak_loader:
        inputs, ancillary_logits = inputs.to(device), ancillary_logits.to(device)

        optimizer.zero_grad()
        primary_logits = primary_model(inputs)

        # Linear blending of logits (self-correction)
        q_logits = (primary_logits + alpha * ancillary_logits) / (1.0 + alpha)

        # Soft pseudo-labels
        q_soft = F.softmax(q_logits.detach(), dim=1)

        # KL divergence between primary logits and soft pseudo-labels
        loss = F.kl_div(
            F.log_softmax(primary_logits, dim=1),
            q_soft,
            reduction='batchmean'
        )
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / (len(fully_labeled_loader) + len(weak_loader))
    print(f"[Epoch {epoch+1}/{num_epochs}] Primary Loss: {avg_loss:.4f}")

# -----------------------
# Save Trained Model
# -----------------------
torch.save(primary_model.state_dict(), os.path.join(save_dir, "primary_model.pth"))
print(f"\n✅ Primary model saved at {os.path.join(save_dir, 'primary_model.pth')}")
