import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Model checkpoint paths
MODEL_PATHS = {
    'Information Matching': 'runs/information_matching_cifar100_experiment/checkpoints/best_model.pth',
    'Label Smoothing': 'runs/resnet18_cifar100_soft_labels/checkpoints/best_model.pth',
    'Standard (Baseline)': 'runs/resnet18_cifar100_baseline/checkpoints/best_model.pth'
}

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# CIFAR-100 classes
cifar100_classes = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]

# CIFAR-100 test loader
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

class ModelEvaluator:
    def __init__(self):
        self.models = {}
        self.model_info = {}
        self.load_models()
        
    def load_models(self):
        """Load all trained models"""
        for name, path in MODEL_PATHS.items():
            try:
                model = torchvision.models.resnet18(weights=None, num_classes=100)
                
                # Modify for CIFAR-100
                model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                model.maxpool = torch.nn.Identity()
                
                # Load checkpoint
                checkpoint = torch.load(path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                
                model.to(device)
                model.eval()
                
                self.models[name] = model
                self.model_info[name] = {
                    'epoch': checkpoint.get('epoch', 'N/A'),
                    'best_val_acc': checkpoint.get('best_val_acc', 'N/A'),
                    'path': path
                }
                print(f"Loaded {name}: Epoch {self.model_info[name]['epoch']}, Val Acc: {self.model_info[name]['best_val_acc']:.2f}%")
            except FileNotFoundError:
                print(f"Warning: Model file not found for {name} at {path}")
            except Exception as e:
                print(f"Error loading {name}: {e}")
        
        if not self.models:
            print("No models could be loaded. Please ensure the model checkpoints exist.")
            exit(1)
    
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
    
    def get_class_images(self, class_idx, num_images=5):
        """Get specific number of images for a given class"""
        class_images = []
        class_labels = []
        
        for imgs, lbls in testloader:
            mask = lbls == class_idx
            if mask.any():
                class_images.append(imgs[mask])
                class_labels.append(lbls[mask])
                
                # Check if we have enough images
                total_images = sum(len(batch) for batch in class_images)
                if total_images >= num_images:
                    break
        
        if class_images:
            all_images = torch.cat(class_images)
            return all_images[:num_images]
        return None
    
    def denormalize_image(self, img):
        """Denormalize CIFAR-100 image for visualization"""
        mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
        std = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)
        return torch.clamp(img * std + mean, 0, 1)
    
    def evaluate_confidence_sweep(self, class_idx=0, num_steps=100, num_images=5):
        """Sweep through distortion levels and evaluate confidence"""
        class_name = cifar100_classes[class_idx]
        print(f"\nEvaluating class {class_idx}: {class_name}")
        
        # Get test images for this class
        class_images = self.get_class_images(class_idx, num_images)
        if class_images is None:
            print(f"No images found for class {class_idx}")
            return
        
        class_images = class_images.to(device)
        print(f"Using {len(class_images)} images for analysis")
        
        # Distortion types and their display names
        distortion_types = ['diffusion', 'blur', 'dropout', 'mosaic']
        distortion_names = {
            'diffusion': 'Diffusion Noise',
            'blur': 'Gaussian Blur',
            'dropout': 'Patch Dropout',
            'mosaic': 'Mosaic Effect'
        }
        
        # Create distortion levels
        d_levels = np.linspace(0, 1, num_steps)
        
        # For each distortion type, create comprehensive analysis
        for dist_type in distortion_types:
            print(f"\nAnalyzing {distortion_names[dist_type]}...")
            
            # Store confidence results for all models and all images
            all_confidences = {name: [] for name in self.models.keys()}
            
            # Store intermediate images for visualization (every 20th step)
            visual_steps = [0, 20, 40, 60, 80, 99]
            visual_images = {step: [] for step in visual_steps}
            
            for step_idx, d in enumerate(tqdm(d_levels, desc=f"Processing {dist_type}")):
                step_confidences = {name: [] for name in self.models.keys()}
                
                # Apply distortion to all images
                distorted_images = torch.stack([
                    self.apply_distortion(img, dist_type, d) for img in class_images
                ])
                
                # Store visual examples
                if step_idx in visual_steps:
                    visual_images[step_idx] = distorted_images.clone()
                
                # Evaluate each model
                for name, model in self.models.items():
                    with torch.no_grad():
                        logits = model(distorted_images)
                        probs = F.softmax(logits, dim=1)
                        class_confs = probs[:, class_idx]
                        step_confidences[name] = class_confs.cpu().numpy()
                
                # Store results
                for name in self.models.keys():
                    all_confidences[name].append(step_confidences[name])
            
            # Convert to numpy arrays
            for name in self.models.keys():
                all_confidences[name] = np.array(all_confidences[name])  # Shape: (steps, num_images)
            
            # Create comprehensive visualization
            self.create_confidence_visualization(
                all_confidences, visual_images, d_levels, visual_steps,
                class_idx, class_name, dist_type, distortion_names[dist_type]
            )
    
    def create_confidence_visualization(self, all_confidences, visual_images, d_levels, 
                                      visual_steps, class_idx, class_name, dist_type, dist_name):
        """Create comprehensive confidence visualization with intermediate images"""
        
        # Create figure with subplots
        plt.figure(figsize=(20, 12))
        
        # Colors for models
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        model_names = list(self.models.keys())
        
        # Top section: Confidence curves for each image individually
        for img_idx in range(len(visual_images[0])):
            ax = plt.subplot(3, 5, img_idx + 1)
            
            for model_idx, name in enumerate(model_names):
                confidences = all_confidences[name][:, img_idx]
                ax.plot(d_levels, confidences, label=name, color=colors[model_idx], 
                       linewidth=2, alpha=0.8)
            
            ax.set_title(f'Image {img_idx + 1}', fontweight='bold', fontsize=12)
            ax.set_xlabel('Distortion Level')
            ax.set_ylabel('Confidence')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            if img_idx == 0:
                ax.legend(loc='upper right', fontsize=10)
        
        # Middle section: Average confidence curves
        ax_avg = plt.subplot(3, 2, 3)
        for model_idx, name in enumerate(model_names):
            avg_confidences = np.mean(all_confidences[name], axis=1)
            std_confidences = np.std(all_confidences[name], axis=1)
            
            ax_avg.plot(d_levels, avg_confidences, label=name, color=colors[model_idx], 
                       linewidth=3)
            ax_avg.fill_between(d_levels, 
                               avg_confidences - std_confidences,
                               avg_confidences + std_confidences,
                               color=colors[model_idx], alpha=0.2)
        
        ax_avg.set_title(f'Average Confidence vs {dist_name}\nClass: {class_name.title()}', 
                        fontweight='bold', fontsize=14)
        ax_avg.set_xlabel('Distortion Level', fontsize=12)
        ax_avg.set_ylabel('Average Confidence', fontsize=12)
        ax_avg.legend(fontsize=11)
        ax_avg.grid(True, alpha=0.3)
        ax_avg.set_xlim(0, 1)
        ax_avg.set_ylim(0, 1)
        
        # Middle section: Confidence degradation rate
        ax_rate = plt.subplot(3, 2, 4)
        for model_idx, name in enumerate(model_names):
            avg_confidences = np.mean(all_confidences[name], axis=1)
            # Calculate degradation rate (negative slope)
            degradation_rate = -np.gradient(avg_confidences, d_levels)
            ax_rate.plot(d_levels, degradation_rate, label=name, color=colors[model_idx], 
                        linewidth=3)
        
        ax_rate.set_title('Confidence Degradation Rate', fontweight='bold', fontsize=14)
        ax_rate.set_xlabel('Distortion Level', fontsize=12)
        ax_rate.set_ylabel('Degradation Rate', fontsize=12)
        ax_rate.legend(fontsize=11)
        ax_rate.grid(True, alpha=0.3)
        ax_rate.set_xlim(0, 1)
        
        # Bottom section: Visual progression of distortion
        for i, step in enumerate(visual_steps):
            ax_img = plt.subplot(3, 6, 13 + i)
            
            # Show first image at this distortion level
            img = visual_images[step][0]  # First image
            img_denorm = self.denormalize_image(img.cpu())
            img_np = img_denorm.permute(1, 2, 0).numpy()
            
            ax_img.imshow(img_np)
            ax_img.set_title(f'd={d_levels[step]:.2f}', fontsize=10)
            ax_img.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'confidence_analysis_{dist_type}_class{class_idx}_{class_name}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: confidence_analysis_{dist_type}_class{class_idx}_{class_name}.png")
        
        # Create a separate detailed heatmap showing all confidences
        self.create_confidence_heatmap(all_confidences, d_levels, class_idx, class_name, 
                                     dist_type, dist_name)
    
    def create_confidence_heatmap(self, all_confidences, d_levels, class_idx, class_name, 
                                dist_type, dist_name):
        """Create heatmap showing confidence variation across all images and distortion levels"""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        model_names = list(self.models.keys())
        
        for model_idx, name in enumerate(model_names):
            ax = axes[model_idx]
            confidences = all_confidences[name].T  # Shape: (num_images, steps)
            
            im = ax.imshow(confidences, aspect='auto', cmap='viridis', 
                          extent=[0, 1, len(confidences), 0])
            ax.set_title(f'{name}\nConfidence Heatmap', fontweight='bold', fontsize=12)
            ax.set_xlabel('Distortion Level')
            ax.set_ylabel('Image Index')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Confidence', rotation=270, labelpad=15)
        
        plt.suptitle(f'Confidence Heatmaps: {dist_name}\nClass: {class_name.title()}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'confidence_heatmap_{dist_type}_class{class_idx}_{class_name}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: confidence_heatmap_{dist_type}_class{class_idx}_{class_name}.png")

def main():
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Choose a visually interesting class (apple is good for testing)
    class_to_analyze = 0  # Apple
    
    print("Starting comprehensive confidence analysis...")
    print(f"Target class: {class_to_analyze} ({cifar100_classes[class_to_analyze]})")
    print(f"Models loaded: {list(evaluator.models.keys())}")
    
    # Run the comprehensive evaluation
    evaluator.evaluate_confidence_sweep(
        class_idx=class_to_analyze,
        num_steps=100,  # 100 distortion levels
        num_images=5    # 5 example images
    )
    
    print("\nAnalysis complete! Check the generated visualization files.")

if __name__ == "__main__":
    main()
