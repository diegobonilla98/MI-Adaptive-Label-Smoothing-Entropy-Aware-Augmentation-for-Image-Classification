import os

import numpy as np
import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class DistortedCIFARDataset(Dataset):
    def __init__(
        self, root_dir="./data", split="train", download=True, use_distortion=False
    ):
        """
        CIFAR-100 Dataset with optional distortions and soft labels based on MI

        Args:
            root_dir (str): Path to store CIFAR-100 data
            split (str): Either 'train' or 'test'
            download (bool): Whether to download CIFAR-100 if not present
            use_distortion: Whether to apply random distortions and soft labels
        """
        self.root_dir = root_dir
        self.split = split
        self.use_distortion = use_distortion
        self.num_classes = 100  # CIFAR-100 has 100 classes

        # Base transforms for CIFAR-10
        base_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        # Load CIFAR-100 dataset
        is_train = split == "train"
        self.cifar_dataset = datasets.CIFAR100(
            root=root_dir, train=is_train, download=download, transform=base_transform
        )

        # CIFAR-100 has 100 fine-grained classes
        # We'll get the class names from the dataset
        self.class_names = self.cifar_dataset.classes

        # Normalization for CIFAR-100 (same as CIFAR-10)
        self.normalize = transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
        )

    def __len__(self):
        return len(self.cifar_dataset)

    def __getitem__(self, idx):
        image, label = self.cifar_dataset[idx]

        if self.use_distortion:
            dist_type = np.random.choice(["diffusion", "blur", "dropout", "mosaic"])
            d = np.random.uniform(0, 1)
            original = image.clone()
            image = self.apply_distortion(image, dist_type, d)
            mi_norm = self.compute_normalized_mi(original, image)
            mi_norm = torch.clamp(torch.tensor(mi_norm), 0.0, 1.0).item()
            epsilon = 1.0 - mi_norm
            epsilon = min(max(epsilon, 0.0), 0.9)
            target = torch.full((self.num_classes,), epsilon / self.num_classes)
            target[label] += 1.0 - epsilon
        else:
            target = torch.zeros(self.num_classes)
            target[label] = 1.0

        # Apply normalization
        image = self.normalize(image)

        return image, target

    def apply_distortion(self, img, dist_type, d):
        """Apply distortion based on type and level d (0: original, 1: fully deformed)"""
        device = img.device
        if dist_type == "diffusion":
            T = 1000
            betas = torch.linspace(1e-4, 0.02, T, device=device)
            alphas = 1.0 - betas
            alpha_bars = torch.cumprod(alphas, dim=0)
            idx = min(max(int(d * (T - 1)), 0), T - 1)
            if idx == 0:
                return img
            sqrt_ab = torch.sqrt(alpha_bars[idx])
            sqrt_1m = torch.sqrt(1 - alpha_bars[idx])
            img_diff = img * 2 - 1
            eps = torch.randn_like(img_diff)
            x_t = sqrt_ab * img_diff + sqrt_1m * eps
            img_t = (x_t + 1) / 2
            return torch.clamp(img_t, 0, 1)

        elif dist_type == "blur":
            max_sigma = 50.0
            sigma = d * max_sigma + 1e-2
            kernel_size = 2 * int(4 * sigma + 0.5) + 1
            kernel_size = min(kernel_size, 51)
            if kernel_size % 2 == 0:
                kernel_size += 1
            return TF.gaussian_blur(
                img, kernel_size=[kernel_size, kernel_size], sigma=sigma
            )

        elif dist_type == "dropout":
            patch_size = 8
            _, H, W = img.shape
            n_h = (H + patch_size - 1) // patch_size
            n_w = (W + patch_size - 1) // patch_size
            total_patches = n_h * n_w
            num_reveal = int(total_patches * (1 - d))
            patch_indices = torch.randperm(total_patches, device=device)[:num_reveal]
            img_t = torch.zeros_like(img)
            for p_idx in patch_indices:
                row = int(p_idx // n_w) * patch_size
                col = int(p_idx % n_w) * patch_size
                row_end = min(row + patch_size, H)
                col_end = min(col + patch_size, W)
                img_t[:, row:row_end, col:col_end] = img[:, row:row_end, col:col_end]
            return img_t

        elif dist_type == "mosaic":
            _, H, W = img.shape
            max_block = max(H, W)
            block_size = max(1, int(max_block * (d + 1e-3) ** 2))
            img_t = torch.zeros_like(img)
            for y in range(0, H, block_size):
                for x in range(0, W, block_size):
                    y_end = min(y + block_size, H)
                    x_end = min(x + block_size, W)
                    block_mean = img[:, y:y_end, x:x_end].mean(dim=(1, 2)).view(3, 1, 1)
                    img_t[:, y:y_end, x:x_end] = block_mean.expand(
                        -1, y_end - y, x_end - x
                    )
            return img_t

        return img

    def compute_normalized_mi(self, img1, img2, bins=64):
        """Compute normalized mutual information between two images using Torch for speed"""
        nmi = 0.0
        eps = 1e-10
        for c in range(3):
            ch1 = img1[c].flatten()
            ch2 = img2[c].flatten()
            bin1 = (ch1 * (bins - 1)).long()
            bin2 = (ch2 * (bins - 1)).long()
            joint_idx = bin1 * bins + bin2
            joint_hist = (
                torch.bincount(joint_idx, minlength=bins**2).float().view(bins, bins)
            )
            joint_prob = joint_hist / (joint_hist.sum() + eps)
            p1 = joint_prob.sum(dim=1)
            p2 = joint_prob.sum(dim=0)
            outer = p1[:, None] * p2[None, :]
            log_term = torch.log2(joint_prob / (outer + eps) + eps)
            mask = joint_prob > 0
            mi_c = (joint_prob[mask] * log_term[mask]).sum().item()
            h1 = -(p1[p1 > 0] * torch.log2(p1[p1 > 0] + eps)).sum().item()
            h2 = -(p2[p2 > 0] * torch.log2(p2[p2 > 0] + eps)).sum().item()
            nmi_c = 2 * mi_c / (h1 + h2) if (h1 + h2) > 0 else 0.0
            nmi += nmi_c
        return nmi / 3


def create_data_loaders(root_dir="./data", batch_size=32, num_workers=4, download=True):
    """Create train and validation data loaders for CIFAR-100"""

    train_dataset = DistortedCIFARDataset(
        root_dir, "train", download=download, use_distortion=True
    )
    val_dataset = DistortedCIFARDataset(
        root_dir, "test", download=download, use_distortion=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loaders - reduced batch size for CIFAR-100
    train_loader, val_loader = create_data_loaders(batch_size=128, num_workers=4)

    # Create model - ResNet18 for CIFAR-100 (100 classes)
    model = models.resnet18(pretrained=False, num_classes=100)

    # Modify first convolution layer for CIFAR-100 (32x32 images)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()  # Remove maxpool for smaller images

    model.to(device)

    # Optimizer with slightly lower learning rate for CIFAR-100
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)

    # Learning rate scheduler for better convergence on CIFAR-100
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    # TensorBoard writer
    writer = SummaryWriter("runs/information_matching_cifar100_experiment")

    # Create checkpoints directory
    checkpoint_dir = "runs/information_matching_cifar100_experiment/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize best validation accuracy for model checkpointing
    best_val_acc = 0.0

    epochs = 100  # Increased for CIFAR-100 (more complex dataset than CIFAR-10)

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training loop with tqdm
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch_idx, (images, targets) in enumerate(train_pbar):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(images)
            log_probs = torch.log_softmax(logits, dim=1)
            loss = -(targets * log_probs).sum(dim=1).mean()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate training accuracy
            _, predicted = logits.max(1)
            _, labels = targets.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(labels).sum().item()

            # Update progress bar with current metrics
            current_loss = running_loss / (batch_idx + 1)
            current_acc = 100.0 * correct_train / total_train
            train_pbar.set_postfix(
                {"Loss": f"{current_loss:.4f}", "Acc": f"{current_acc:.2f}%"}
            )

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct_train / total_train

        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%"
        )

        # Log training metrics to TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)

                logits = model(images)
                log_probs = torch.log_softmax(logits, dim=1)
                loss = -(targets * log_probs).sum(dim=1).mean()
                val_loss += loss.item()

                _, predicted = logits.max(1)
                _, labels = targets.max(1)
                total += targets.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100.0 * correct / total
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

        # Log validation metrics to TensorBoard
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_acc": best_val_acc,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                },
                checkpoint_path,
            )
            print(f"New best model saved with validation accuracy: {best_val_acc:.2f}%")

        # Step the learning rate scheduler
        scheduler.step()

        # Log learning rate
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], epoch)

    writer.close()
    print(
        "Training completed. TensorBoard logs saved to 'runs/information_matching_cifar100_experiment'"
    )
    print(
        f"Best model saved to '{checkpoint_dir}/best_model.pth' with validation accuracy: {best_val_acc:.2f}%"
    )
    print("To view logs, run: tensorboard --logdir=runs")


if __name__ == "__main__":
    train_model()
