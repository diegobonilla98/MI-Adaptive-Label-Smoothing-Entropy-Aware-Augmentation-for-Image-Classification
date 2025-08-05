import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# Model checkpoint paths based on the training scripts
MODEL_PATHS = {
    "Information Matching": "runs/information_matching_cifar100_experiment/checkpoints/best_model.pth",
    "Label Smoothing": "runs/resnet18_cifar100_soft_labels/checkpoints/best_model.pth",
    "Standard (Baseline)": "runs/resnet18_cifar100_baseline/checkpoints/best_model.pth",
}

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-100 test loader (unseen set)
transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        ),  # CIFAR-100 means/std
    ]
)
testset = torchvision.datasets.CIFAR100(
    root="./data", train=False, download=True, transform=transform_test
)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)


# Function to load models from checkpoints
def load_model(path):
    model = torchvision.models.resnet18(weights=None, num_classes=100)

    # Modify first convolution layer for CIFAR-100 (32x32 images) - same as training
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()  # Remove maxpool for smaller images

    # Load checkpoint
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()
    return model, checkpoint


# Load all models
models = {}
model_info = {}
for name, path in MODEL_PATHS.items():
    try:
        model, checkpoint = load_model(path)
        models[name] = model
        model_info[name] = {
            "epoch": checkpoint.get("epoch", "N/A"),
            "best_val_acc": checkpoint.get("best_val_acc", "N/A"),
            "path": path,
        }
        print(
            f"Loaded {name}: Epoch {model_info[name]['epoch']}, Val Acc: {model_info[name]['best_val_acc']:.2f}%"
        )
    except FileNotFoundError:
        print(f"Warning: Model file not found for {name} at {path}")
        print(f"Skipping {name} from evaluation.")
    except Exception as e:
        print(f"Error loading {name}: {e}")
        print(f"Skipping {name} from evaluation.")

if not models:
    print("No models could be loaded. Please ensure the model checkpoints exist.")
    exit(1)


# Distortion functions (adapted from previous code, for CIFAR 32x32)
def apply_distortion(img, dist_type, d):
    """Apply distortion based on type and level d (0: original, 1: fully deformed)"""
    # Handle both 3D (C, H, W) and 4D (B, C, H, W) tensors
    if len(img.shape) == 4:
        # Batch dimension present, process each image in batch
        return torch.stack(
            [apply_distortion(single_img, dist_type, d) for single_img in img]
        )

    if dist_type == "diffusion":
        T = 1000
        betas = torch.linspace(1e-4, 0.02, T, device=img.device)
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
        max_sigma = 10.0  # Smaller for 32x32 images
        sigma = d * max_sigma + 1e-2
        kernel_size = 2 * int(4 * sigma + 0.5) + 1
        kernel_size = min(kernel_size, 11)  # Cap for small images
        if kernel_size % 2 == 0:
            kernel_size += 1
        return transforms.functional.gaussian_blur(
            img, kernel_size=[kernel_size, kernel_size], sigma=sigma
        )

    elif dist_type == "dropout":
        patch_size = 4  # Smaller patches for 32x32
        _, H, W = img.shape
        n_h = (H + patch_size - 1) // patch_size
        n_w = (W + patch_size - 1) // patch_size
        total_patches = n_h * n_w
        num_reveal = int(total_patches * (1 - d))
        patch_indices = torch.randperm(total_patches, device=img.device)[:num_reveal]
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
                img_t[:, y:y_end, x:x_end] = block_mean.expand(-1, y_end - y, x_end - x)
        return img_t

    return img


# Evaluation functions


def compute_accuracy(model, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def compute_top5_accuracy(model, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = outputs.topk(5, 1, True, True)
            correct += (
                predicted.eq(labels.view(-1, 1).expand_as(predicted)).sum().item()
            )
            total += labels.size(0)
    return 100 * correct / total


def compute_loss(model, loader):
    loss = 0
    n = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            n += 1
    return loss / n


def compute_ece(model, loader, n_bins=10):
    """Expected Calibration Error"""
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    confidences_list = []
    accuracies_list = []
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            confs, preds = probs.max(1)
            correctness = preds.eq(labels)
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (confs > bin_lower) & (confs <= bin_upper)
                if in_bin.any():
                    bin_acc = correctness[in_bin].float().mean().item()
                    bin_conf = confs[in_bin].mean().item()
                    accuracies_list.append(bin_acc)
                    confidences_list.append(bin_conf)
    ece = 0
    for acc, conf in zip(accuracies_list, confidences_list):
        ece += abs(acc - conf) / len(accuracies_list)
    return ece


# Robustness: Accuracy under random degradations
def compute_robust_accuracy(model, loader, dist_types, num_levels=5):
    accs = {dt: [] for dt in dist_types}
    for dist_type in dist_types:
        for level in np.linspace(0, 1, num_levels):
            correct = 0
            total = 0
            with torch.no_grad():
                for data in loader:
                    images, labels = data[0].to(device), data[1].to(device)
                    degraded = torch.stack(
                        [apply_distortion(img, dist_type, level) for img in images]
                    )
                    outputs = model(degraded)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accs[dist_type].append(100 * correct / total)
    return accs


# Run general evaluations
print("\nRunning evaluations...")
dist_types = ["diffusion", "blur", "dropout", "mosaic"]
results = {}
for name, model in models.items():
    print(f"Evaluating {name}...")
    results[name] = {
        "top1_acc": compute_accuracy(model, testloader),
        "top5_acc": compute_top5_accuracy(model, testloader),
        "loss": compute_loss(model, testloader),
        "ece": compute_ece(model, testloader),
        "robust_acc": compute_robust_accuracy(model, testloader, dist_types),
    }
    print(f"  Top-1 Accuracy: {results[name]['top1_acc']:.2f}%")
    print(f"  Top-5 Accuracy: {results[name]['top5_acc']:.2f}%")

# Plot general metrics with improved styling
metrics = ["top1_acc", "top5_acc", "loss", "ece"]
metric_labels = [
    "Top-1 Accuracy (%)",
    "Top-5 Accuracy (%)",
    "Cross-Entropy Loss",
    "Expected Calibration Error",
]
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, Orange, Green

for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
    ax = axs[i // 2, i % 2]
    values = [results[name][metric] for name in models]
    bars = ax.bar(range(len(models)), values, color=colors[: len(models)])
    ax.set_title(f"{label} on Clean CIFAR-100 Test Set", fontsize=14, fontweight="bold")
    ax.set_ylabel(label, fontsize=12)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models.keys(), rotation=45, ha="right")
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + height * 0.01,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

plt.tight_layout()
plt.savefig("general_metrics_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: general_metrics_comparison.png")

# Robustness plots: joined per distortion type with improved styling
distortion_names = {
    "diffusion": "Diffusion Noise",
    "blur": "Gaussian Blur",
    "dropout": "Patch Dropout",
    "mosaic": "Mosaic Effect",
}

for dist_type in dist_types:
    fig, ax = plt.subplots(figsize=(10, 7))
    levels = np.linspace(0, 1, 5)

    for i, name in enumerate(models):
        ax.plot(
            levels,
            results[name]["robust_acc"][dist_type],
            label=name,
            marker="o",
            linewidth=2.5,
            markersize=6,
            color=colors[i],
        )

    ax.set_title(
        f"Model Robustness: Accuracy vs. {distortion_names[dist_type]} Level",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("Distortion Level (0 = Original, 1 = Maximum)", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(
        0, max([max(results[name]["robust_acc"][dist_type]) for name in models]) + 5
    )

    plt.tight_layout()
    plt.savefig(f"robust_acc_vs_{dist_type}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: robust_acc_vs_{dist_type}.png")

# Individual robustness per model with improved styling
dist_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]  # Red, Blue, Green, Orange

for name in models:
    fig, ax = plt.subplots(figsize=(10, 7))
    levels = np.linspace(0, 1, 5)

    for i, dist_type in enumerate(dist_types):
        ax.plot(
            levels,
            results[name]["robust_acc"][dist_type],
            label=distortion_names[dist_type],
            marker="s",
            linewidth=2.5,
            markersize=6,
            color=dist_colors[i],
        )

    ax.set_title(f"Robustness Analysis: {name} Model", fontsize=16, fontweight="bold")
    ax.set_xlabel("Distortion Level (0 = Original, 1 = Maximum)", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(
        f'robust_acc_per_dist_{name.replace(" ", "_").replace("(", "").replace(")", "").lower()}.png',
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(
        f"Saved: robust_acc_per_dist_{name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.png"
    )

# Specific test: Confidence vs. distortion for a single class
print("\nAnalyzing confidence degradation for specific class...")
# Pick class 0 (apple in CIFAR-100), get all test images for it
class_idx = 0
cifar100_classes = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
]  # First 10 for reference
class_name = (
    cifar100_classes[class_idx]
    if class_idx < len(cifar100_classes)
    else f"class_{class_idx}"
)

class_images = []
class_labels = []
for imgs, lbls in testloader:
    mask = lbls == class_idx
    if mask.any():
        class_images.append(imgs[mask])
        class_labels.append(lbls[mask])

if class_images:
    class_images = torch.cat(class_images).to(device)
    num_images = len(class_images)
    print(f"Using {num_images} images for class {class_idx} ({class_name})")

    # Steps for distortion
    d_steps = np.linspace(0, 1, 11)  # 0 to 1 in 0.1 steps

    # Compute avg confidence per model, per distortion type, per distortion level
    conf_results = {name: {dt: [] for dt in dist_types} for name in models}
    for name, model in models.items():
        print(f"Computing confidence for {name}...")
        for dist_type in dist_types:
            for d in tqdm(
                d_steps, desc=f"{name} - {distortion_names[dist_type]}", leave=False
            ):
                degraded = torch.stack(
                    [apply_distortion(img, dist_type, d) for img in class_images]
                )
                with torch.no_grad():
                    logits = model(degraded)
                    probs = F.softmax(logits, dim=1)
                    class_confs = probs[:, class_idx]
                    avg_conf = class_confs.mean().item()
                conf_results[name][dist_type].append(avg_conf)
else:
    print(f"No images found for class {class_idx}. Skipping confidence analysis.")
    conf_results = None

# Plot confidence vs. distortion level: joined per distortion type
if conf_results:
    for dist_type in dist_types:
        fig, ax = plt.subplots(figsize=(10, 7))

        for i, name in enumerate(models):
            ax.plot(
                d_steps,
                conf_results[name][dist_type],
                label=name,
                marker="o",
                linewidth=2.5,
                markersize=6,
                color=colors[i],
            )

        ax.set_title(
            f"Model Confidence vs. {distortion_names[dist_type]} Level\n(True Class: {class_name.title()})",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_xlabel("Distortion Level (0 = Original, 1 = Maximum)", fontsize=13)
        ax.set_ylabel("Average Confidence for True Class", fontsize=13)
        ax.legend(fontsize=11, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(
            f"confidence_vs_{dist_type}_class{class_idx}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print(f"Saved: confidence_vs_{dist_type}_class{class_idx}.png")

    # Individual confidence plots per model
    for name in models:
        fig, ax = plt.subplots(figsize=(10, 7))

        for i, dist_type in enumerate(dist_types):
            ax.plot(
                d_steps,
                conf_results[name][dist_type],
                label=distortion_names[dist_type],
                marker="s",
                linewidth=2.5,
                markersize=6,
                color=dist_colors[i],
            )

        ax.set_title(
            f"Confidence Analysis: {name} Model\n(True Class: {class_name.title()})",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_xlabel("Distortion Level (0 = Original, 1 = Maximum)", fontsize=13)
        ax.set_ylabel("Average Confidence for True Class", fontsize=13)
        ax.legend(fontsize=11, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        filename = f'confidence_per_dist_class{class_idx}_{name.replace(" ", "_").replace("(", "").replace(")", "").lower()}.png'
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {filename}")

print("\nAll evaluations completed and plots saved!")
print("\nSummary of Results:")
print("=" * 50)
for name in models:
    print(f"\n{name}:")
    print(
        f"  Training Info: Epoch {model_info[name]['epoch']}, Val Acc: {model_info[name]['best_val_acc']:.2f}%"
    )
    print(f"  Test Top-1 Accuracy: {results[name]['top1_acc']:.2f}%")
    print(f"  Test Top-5 Accuracy: {results[name]['top5_acc']:.2f}%")
    print(f"  Expected Calibration Error: {results[name]['ece']:.4f}")
    print(f"  Cross-Entropy Loss: {results[name]['loss']:.4f}")
