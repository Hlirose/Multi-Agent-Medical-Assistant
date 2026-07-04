import os
import cv2
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


class BrainTumorUNet(nn.Module):
    """U-Net model for brain tumor segmentation in MRI scans."""

    def __init__(self, n_channels=1, n_classes=1):
        super(BrainTumorUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.enc1 = self._conv_block(n_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        self.bottleneck = self._conv_block(512, 1024)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128, 64)

        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.upconv4(b)
        d4 = torch.cat([e4, d4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.upconv3(d4)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)


class BrainTumorSegmentation:
    """Handles brain tumor segmentation in MRI scans using a trained U-Net model."""

    def __init__(self, model_path):
        self.model_path = model_path
        self.device = DEVICE
        self.img_size = 256
        self.model = self._load_model()

    def _load_model(self):
        model = BrainTumorUNet(n_channels=1, n_classes=1).to(self.device)
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                logger.info(f"Brain tumor model loaded from {self.model_path}")
            except Exception as e:
                logger.warning(f"Could not load brain tumor model weights: {e}")
                logger.warning("Using randomly initialized weights. Train or provide a checkpoint for real predictions.")
        else:
            logger.warning(f"Model file not found at {self.model_path}")
            logger.warning("Using randomly initialized weights. Train or provide a checkpoint for real predictions.")

        model.eval()
        return model

    def _preprocess(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")

        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0).to(self.device)
        return img_tensor, img

    def _overlay_segmentation(self, original_img, mask, output_path):
        original_resized = cv2.resize(original_img, (mask.shape[1], mask.shape[0]))
        original_rgb = cv2.cvtColor(original_resized, cv2.COLOR_GRAY2RGB)

        colored_mask = np.zeros_like(original_rgb)
        colored_mask[mask > 0.5] = [255, 0, 0]

        overlay = cv2.addWeighted(original_rgb, 0.6, colored_mask, 0.4, 0)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original_resized, cmap='gray')
        axes[0].set_title('Original MRI')
        axes[0].axis('off')
        axes[1].imshow(mask, cmap='hot')
        axes[1].set_title('Tumor Segmentation')
        axes[1].axis('off')
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        logger.info(f"Segmentation result saved to {output_path}")
        return True

    def predict(self, image_path, output_path):
        """
        Segment brain tumor in MRI image.

        Args:
            image_path: Path to the MRI image
            output_path: Path to save the segmentation visualization

        Returns:
            dict with 'has_tumor' (bool), 'tumor_area_ratio' (float), 'output_path' (str)
        """
        try:
            img_tensor, original_img = self._preprocess(image_path)

            with torch.no_grad():
                output = self.model(img_tensor)
                mask = torch.sigmoid(output).squeeze().cpu().numpy()

            mask_resized = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]))

            tumor_area_ratio = float(np.sum(mask_resized > 0.5) / mask_resized.size)
            has_tumor = tumor_area_ratio > 0.01

            self._overlay_segmentation(original_img, mask_resized, output_path)

            result = {
                "has_tumor": has_tumor,
                "tumor_area_ratio": round(tumor_area_ratio, 4),
                "output_path": output_path,
            }

            logger.info(f"Prediction result: has_tumor={has_tumor}, area_ratio={tumor_area_ratio:.4f}")
            return result

        except Exception as e:
            logger.error(f"Error during brain tumor segmentation: {e}")
            return None