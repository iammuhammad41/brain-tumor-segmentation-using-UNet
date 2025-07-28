# **BraTS 2024 CGAN-U-Net for Brain Tumor Segmentation**

This repository contains the code for training a **Conditional Generative Adversarial Network (CGAN)** with a **U-Net** generator for brain tumor segmentation. The model is trained on the **BraTS 2024 dataset** to predict tumor segments and enhance medical imaging segmentation tasks. The code also includes data preprocessing, augmentation, and evaluation using common image quality metrics.

## **Key Features**

* **Conditional GAN (CGAN)** architecture with a **U-Net** generator and a **Patch Discriminator**.
* Preprocessing of the **BraTS 2024** dataset for brain tumor segmentation.
* Evaluation of model performance using metrics such as **PSNR** (Peak Signal-to-Noise Ratio) and **SSIM** (Structural Similarity Index).
* Training the model for **100** epochs (with checkpoints) and saving the model weights for later use.

## **Dataset**

This project uses the **BraTS 2024** dataset for brain tumor segmentation. The dataset consists of multimodal MRI images (T1, T1c, T2w, T2f) for each patient, along with the corresponding **segmentation masks**.

### **Dataset Structure**

The dataset is organized as follows:

```
/BraTS2024_small_dataset/
    ├── patient_1/
    │   ├── t1n.nii
    │   ├── t1c.nii
    │   ├── t2w.nii
    │   ├── t2f.nii
    │   ├── seg.nii
    ├── patient_2/
    │   ├── t1n.nii
    │   ├── t1c.nii
    │   ├── t2w.nii
    │   ├── t2f.nii
    │   ├── seg.nii
    └── ...
```

### **Preprocessing**

Each MRI volume is preprocessed by normalizing the pixel values and resizing them to a fixed size (`256x256`). This preprocessing is applied slice by slice for each patient, generating `.npy` files for each modality.

## **Installation**

Ensure you have the necessary dependencies installed:

```bash
pip install torch torchvision matplotlib tqdm scikit-image nibabel opencv-python
```

## **How to Use**

### **1. Dataset Preparation**

To preprocess the **BraTS 2024** dataset and save it as `.npy` files:

```python
import os
import numpy as np
import nibabel as nib
import cv2
from glob import glob

BASE_PATH = "/path/to/BraTS2024_small_dataset"
SAVE_PATH = "/path/to/save/preprocessed_data"
os.makedirs(SAVE_PATH, exist_ok=True)

MODALITIES = {
    "t1n": "t1n.nii",
    "t1c": "t1c.nii",
    "t2w": "t2w.nii",
    "t2f": "t2f.nii",
    "seg": "seg.nii"
}

IMG_SIZE = 256
SLICES_PER_PATIENT = 20

def preprocess_slice(img, size=IMG_SIZE):
    img = np.nan_to_num(img)
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

patients = sorted(glob(os.path.join(BASE_PATH, "*")))

for patient in patients[:50]:  # Reduce if needed
    pid = os.path.basename(patient)
    volumes = {}

    for mod, suffix in MODALITIES.items():
        path = glob(os.path.join(patient, f"*{suffix}"))[0]
        volumes[mod] = nib.load(path).get_fdata()

    center = volumes["t1n"].shape[2] // 2
    z_range = range(center - SLICES_PER_PATIENT//2, center + SLICES_PER_PATIENT//2)

    for z in z_range:
        for mod in MODALITIES:
            img = preprocess_slice(volumes[mod][:, :, z])
            np.save(os.path.join(SAVE_PATH, f"{pid}_z{z}_{mod}.npy"), img)
```

### **2. Training the Model**

To train the **CGAN with U-Net** generator and discriminator:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Define the Generator and Discriminator models (U-Net and Patch Discriminator)

class GeneratorUNet(nn.Module):
    # U-Net Generator Implementation

class PatchDiscriminator(nn.Module):
    # Patch-based Discriminator Implementation

# Instantiate models
generator = GeneratorUNet().to(device)
discriminator = PatchDiscriminator().to(device)

# Define optimizers and loss functions
g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

# Training loop
EPOCHS = 100
losses_g, losses_d, l1s = [], [], []

for epoch in range(1, EPOCHS + 1):
    g_epoch_loss, d_epoch_loss, l1_epoch = 0, 0, 0
    for x, y in tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}"):
        d_loss, g_loss, l1_loss_val = train_step(x, y)
        d_epoch_loss += d_loss
        g_epoch_loss += g_loss
        l1_epoch += l1_loss_val

    print(f"Epoch {epoch} | D: {d_epoch_loss:.4f} | G: {g_epoch_loss:.4f} | L1: {l1_epoch:.4f}")
    losses_d.append(d_epoch_loss)
    losses_g.append(g_epoch_loss)
    l1s.append(l1_epoch)
```

### **3. Evaluation**

Once training is completed, you can evaluate the model performance using **SSIM** and **PSNR**:

```python
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

def evaluate(generator, dataloader):
    generator.eval()
    ssim_total, psnr_total = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            preds = generator(x).cpu().numpy()
            targets = y.numpy()

            preds = np.clip(preds, 0, 1)
            targets = np.clip(targets, 0, 1)

            for i in range(preds.shape[0]):
                psnr_total.append(peak_signal_noise_ratio(targets[i][0], preds[i][0], data_range=1.0))
                ssim_total.append(structural_similarity(targets[i][0], preds[i][0], data_range=1.0))

    return np.mean(ssim_total), np.mean(psnr_total)

ssim_val, psnr_val = evaluate(generator, dataloader)
print(f"SSIM (T2f): {ssim_val:.4f}")
print(f"PSNR (T2f): {psnr_val:.4f}")
```

### **4. Save and Load Model Checkpoints**

During training, the model is saved every 10 epochs for later use:

```python
# Save checkpoints
torch.save({
    'generator': generator.state_dict(),
    'discriminator': discriminator.state_dict(),
}, "cgan_models_epoch.pth")

# Load checkpoints
checkpoint = torch.load("cgan_models_epoch.pth", map_location=device)
generator.load_state_dict(checkpoint['generator'])
discriminator.load_state_dict(checkpoint['discriminator'])
```

## **Model Architecture**

* **Generator**: U-Net architecture with skip connections, with input channels for multi-modal MRI and output channels for tumor segmentation.
* **Discriminator**: Patch-based architecture that distinguishes real and fake segments.

## **Evaluation Metrics**

* **SSIM** (Structural Similarity Index) for measuring the similarity between the predicted and ground truth images.
* **PSNR** (Peak Signal-to-Noise Ratio) for evaluating image quality.

## **Results Visualization**

The `visualize_samples` function can be used to visualize the predictions made by the model.

```python
def visualize_samples(generator, dataloader):
    generator.eval()
    x, y = next(iter(dataloader))
    x = x.to(device)
    with torch.no_grad():
        preds = generator(x).cpu().numpy()
    x = x.cpu().numpy()
    y = y.cpu().numpy()

    plt.figure(figsize=(15, 8))
    for i in range(3):
        plt.subplot(3, 5, i*5 + 1)
        plt.imshow(x[i][0], cmap='gray')
        plt.title("Input: T1n")

        plt.subplot(3, 5, i*5 + 2)
        plt.imshow(x[i][1], cmap='gray')
        plt.title("Input: T1c")

        plt.subplot(3, 5, i*5 + 3)
        plt.imshow(x[i][2], cmap='gray')
        plt.title("Input: T2w")

        plt.subplot(3, 5, i*5 + 4)
        plt.imshow(y[i][0], cmap='gray')
        plt.title("GT: T2f")

        plt.subplot(3, 5, i*5 + 5)
        plt.imshow(preds[i][0], cmap='gray')
        plt.title("Pred: T2f")

    plt.tight_layout()
    plt.show()

visualize_samples(generator, dataloader)
```

