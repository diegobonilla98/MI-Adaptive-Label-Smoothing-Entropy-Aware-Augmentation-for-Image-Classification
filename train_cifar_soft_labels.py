import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.models as models
import torchvision.datasets as datasets
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os


class DistortedCIFARDataset(Dataset):
    def __init__(self, root_dir="./data", split="train", download=True, use_distortion=False, label_smoothing=0.1):
        """
        CIFAR-100 Dataset with optional distortions and soft label smoothing
        
        Args:
            root_dir (str): Path to store CIFAR-100 data
            split (str): Either 'train' or 'test'
            download (bool): Whether to download CIFAR-100 if not present
            use_distortion: Whether to apply random distortions
            label_smoothing: Label smoothing factor (0.0 = no smoothing, 0.1 = 10% smoothing)
        """
        self.root_dir = root_dir
        self.split = split
        self.use_distortion = use_distortion
        self.label_smoothing = label_smoothing
        self.num_classes = 100  # CIFAR-100 has 100 classes
        
        # Base transforms for CIFAR-100
        # Add data augmentation for the training set
        if split == "train":
            base_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            base_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        
        # Load CIFAR-100 dataset
        is_train = (split == "train")
        self.cifar_dataset = datasets.CIFAR100(
            root=root_dir, 
            train=is_train, 
            download=download, 
            transform=base_transform
        )
        
        # CIFAR-100 has 100 fine-grained classes
        # We'll get the class names from the dataset
        self.class_names = self.cifar_dataset.classes
        
        # Normalization for CIFAR-100 (same as CIFAR-10)
        self.normalize = transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408], 
            std=[0.2675, 0.2565, 0.2761]
        )
        
    def __len__(self):
        return len(self.cifar_dataset)
    
    def __getitem__(self, idx):
        image, label = self.cifar_dataset[idx]
        
        if self.use_distortion:
            dist_type = np.random.choice(['diffusion', 'blur', 'dropout', 'mosaic'])
            d = np.random.uniform(0, 1)
            image = self.apply_distortion(image, dist_type, d)
        
        # Apply normalization
        image = self.normalize(image)
        
        # Create soft labels with label smoothing
        if self.label_smoothing > 0.0 and self.split == "train":
            # Apply label smoothing: (1 - smoothing) * one_hot + smoothing / num_classes
            target = torch.full((self.num_classes,), self.label_smoothing / self.num_classes)
            target[label] = 1.0 - self.label_smoothing + (self.label_smoothing / self.num_classes)
            return image, target
        else:
            # Return standard integer label for validation or when no smoothing
            return image, label
    
    def apply_distortion(self, img, dist_type, d):
        """Apply distortion based on type and level d (0: original, 1: fully deformed)"""
        device = img.device
        if dist_type == 'diffusion':
            T = 1000
            betas = torch.linspace(1e-4, 0.02, T, device=device)
            alphas = 1. - betas
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
        
        elif dist_type == 'blur':
            max_sigma = 50.0
            sigma = d * max_sigma + 1e-2
            kernel_size = 2 * int(4 * sigma + 0.5) + 1
            kernel_size = min(kernel_size, 51)
            if kernel_size % 2 == 0:
                kernel_size += 1
            return TF.gaussian_blur(img, kernel_size=[kernel_size, kernel_size], sigma=sigma)
        
        elif dist_type == 'dropout':
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
        
        elif dist_type == 'mosaic':
            _, H, W = img.shape
            max_block = max(H, W)
            block_size = max(1, int(max_block * (d + 1e-3)**2))
            img_t = torch.zeros_like(img)
            for y in range(0, H, block_size):
                for x in range(0, W, block_size):
                    y_end = min(y + block_size, H)
                    x_end = min(x + block_size, W)
                    block_mean = img[:, y:y_end, x:x_end].mean(dim=(1, 2)).view(3, 1, 1)
                    img_t[:, y:y_end, x:x_end] = block_mean.expand(-1, y_end - y, x_end - x)
            return img_t
        
        return img


def create_data_loaders(root_dir="./data", batch_size=128, num_workers=4, download=True, label_smoothing=0.1):
    """Create train and validation data loaders for CIFAR-100"""
    
    train_dataset = DistortedCIFARDataset(root_dir, "train", download=download, use_distortion=True, label_smoothing=label_smoothing)
    val_dataset = DistortedCIFARDataset(root_dir, "test", download=download, use_distortion=False, label_smoothing=0.0)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data loaders with label smoothing
    label_smoothing = 0.1  # 10% label smoothing
    train_loader, val_loader = create_data_loaders(batch_size=128, num_workers=4, label_smoothing=label_smoothing)

    # Create model - ResNet18 for CIFAR-100 (100 classes)
    model = models.resnet18(weights=None, num_classes=100)
    
    # Modify first convolution layer for CIFAR-100 (32x32 images)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()  # Remove maxpool for smaller images
    
    model.to(device)
    
    # Loss function - we'll use a custom loss for soft labels in training
    criterion = torch.nn.CrossEntropyLoss()  # For validation (hard labels)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    
    # TensorBoard writer
    writer = SummaryWriter('runs/resnet18_cifar100_soft_labels')
    
    # Create checkpoints directory
    checkpoint_dir = 'runs/resnet18_cifar100_soft_labels/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize best validation accuracy for model checkpointing
    best_val_acc = 0.0
    
    epochs = 100
    
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
            
            # Check if we have soft labels (probability distribution) or hard labels (integer)
            if targets.dim() == 2:  # Soft labels (batch_size, num_classes)
                # Use KL divergence loss for soft labels
                log_probs = torch.log_softmax(logits, dim=1)
                loss = -(targets * log_probs).sum(dim=1).mean()
                # For accuracy calculation, get the true class (highest probability)
                _, labels = targets.max(1)
            else:  # Hard labels (batch_size,)
                loss = criterion(logits, targets)
                labels = targets
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate training accuracy
            _, predicted = logits.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
            
            # Update progress bar with current metrics
            current_loss = running_loss / (batch_idx + 1)
            current_acc = 100. * correct_train / total_train
            train_pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct_train / total_train
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Log training metrics to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        
        # Log validation metrics to TensorBoard
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, checkpoint_path)
            print(f"New best model saved with validation accuracy: {best_val_acc:.2f}%")
        
        # Step the learning rate scheduler
        scheduler.step()
        
        # Log learning rate
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
    
    writer.close()
    print("Training completed. TensorBoard logs saved to 'runs/resnet18_cifar100_soft_labels'")
    print(f"Best model saved to '{checkpoint_dir}/best_model.pth' with validation accuracy: {best_val_acc:.2f}%")
    print("To view logs, run: tensorboard --logdir=runs")

if __name__ == "__main__":
    train_model()
