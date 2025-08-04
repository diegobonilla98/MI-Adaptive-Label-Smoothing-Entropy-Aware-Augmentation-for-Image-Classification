# MI-Adaptive Label Smoothing: Entropy-Aware Augmentation for Image Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project introduces a novel approach to data augmentation in deep learning for image classification. The core idea is to dynamically adjust the class confidence (via soft labels) in the cross-entropy loss based on the normalized mutual information (NMI) between an original image and its degraded (augmented) version. Instead of treating all augmented samples as equally reliable with hard one-hot labels, we soften the labels proportionally to the information loss introduced by degradations like noise, blur, dropout, or mosaic effects. This makes the training process more robust to real-world variations and aligns with information-theoretic principles, potentially improving generalization and calibration.

The implementation is built in PyTorch and tested on subsets like ImageNet-100 (a 100-class subset of ImageNet with ~130,000 training images and ~5,000 validation images). It uses ResNet-18 as the backbone and includes custom distortions that preserve topology while removing information. Future work includes full experiments on CIFAR-100 and comparisons to state-of-the-art (SOTA) methods.

## Inspiration and Intuition

In standard deep learning pipelines, data augmentations (e.g., adding Gaussian noise, blurring, or pixel dropout) are used to enhance model robustness by exposing it to varied inputs. However, these methods typically retain hard labels (e.g., [1, 0, 0, ...]), assuming the augmented image is as informative as the original for classification. This overlooks a key reality: degradations reduce the mutual information shared with the original, increasing uncertainty.

The intuition is rooted in information theory: If an augmentation acts like a noisy channel that erodes details without altering the underlying class (e.g., topology-preserving operations), the model's confidence should scale with the remaining information. By estimating NMI between the original and augmented images, we quantify this "information retention" and use it to soften labels—making the loss less punitive for highly degraded samples. This prevents overconfidence on noisy data, encourages better entropy handling, and bridges the gap between binary labels and real entropy in cross-entropy loss.

Inspiration draws from:
- Label smoothing techniques, which add fixed noise to labels for regularization.
- Contrastive learning (e.g., SimCLR), where MI is maximized between views for invariance.
- Noisy label handling, where adaptive confidence weights mitigate uncertainty.
- Communication theory, viewing augmentations as channels with capacity limits.

This approach ensures training accounts for mathematically defined entropy, potentially leading to models that are more calibrated and robust in low-information scenarios (e.g., blurry or noisy real-world images).

## Mathematics

### Mutual Information and Normalized MI
Mutual information $I(X; Y)$ measures the shared information between two random variables $X$ (original image) and $Y$ (augmented image):

$$
I(X; Y) = \sum_{x,y} p(x,y) \log_2 \left( \frac{p(x,y)}{p(x)p(y)} \right)
$$

Where $p(x,y)$ is the joint probability, and $p(x), p(y)$ are marginals. Entropy $H(X) = -\sum_x p(x) \log_2 p(x)$ quantifies uncertainty.

For images, we approximate via histograms (per RGB channel, 64 bins) and average NMI:

$$
\text{NMI}(X, Y) = \frac{2 \cdot I(X; Y)}{H(X) + H(Y)}
$$

NMI ranges from 0 (independent, full information loss) to 1 (identical). We clamp it to [0,1] for stability.

### Soft Labels and Loss
Let $c = \text{NMI}(X, Y)$, $\epsilon = 1 - c$ (capped at 0.9 to avoid uniform labels). For a K-class problem with true label $l$, the soft target $\mathbf{t}$ is:

$$
t_i = 
\begin{cases} 
1 - \epsilon & \text{if } i = l \\
\frac{\epsilon}{K - 1} & \text{otherwise}
\end{cases}
$$

(Note: The code uses $\frac{\epsilon}{K}$ for simplicity, which is approximate for large K like 100.)

The loss is soft cross-entropy (equivalent to KL divergence):

$$
\mathcal{L} = -\frac{1}{N} \sum_{n=1}^N \sum_{k=1}^K t_{n,k} \log p_{n,k}
$$

Where $p = \softmax(\text{logits})$. This reduces to standard CE for $\epsilon = 0$ (no degradation).

### Distortions as Information-Degrading Operations
We apply topology-preserving augmentations:
- **Diffusion**: Forward DDPM process: $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$, with $t$ scaled by distortion level $d \in [0,1]$.
- **Blur**: Gaussian filter with $\sigma = d \cdot 50$.
- **Dropout**: Patch-based masking, revealing $(1-d)$ fraction of patches.
- **Mosaic**: Block averaging with size $\max(1, \sqrt{d} \cdot \max(H,W))$.

These increase conditional entropy $H(X|Y)$, lowering NMI without flipping classes.

## Implementation

The code is a self-contained PyTorch script for training on ImageNet-100. It defines a custom `Dataset` class to handle on-the-fly distortions and MI-based soft labels, then trains a ResNet-18 model.

### Key Components

#### Dataset: `DistortedImageNet100Dataset`
- Loads ImageNet-100 from a root directory (expects train/val splits and `Labels.json` for class mappings).
- In `__getitem__`:
  - Loads and resizes image to 224x224, converts to tensor.
  - If `use_distortion=True` (training):
    - Clones original tensor.
    - Randomly selects distortion type and level $d \sim U(0,1)$.
    - Applies distortion.
    - Computes NMI using histogram method.
    - Creates soft target based on NMI.
  - Else (validation): Uses hard one-hot target.
- Normalizes with ImageNet stats after distortion.

#### Distortion Functions (`apply_distortion`)
- Implemented as tensor operations for efficiency.
- Diffusion uses a linear beta schedule (1e-4 to 0.02 over 1000 steps).
- Blur uses `torchvision.functional.gaussian_blur`.
- Dropout zeros random 8x8 patches.
- Mosaic averages blocks and expands.

#### MI Computation (`compute_normalized_mi`)
- Flattens channels, bins to 64 levels.
- Computes joint/marginal histograms with `bincount`.
- Calculates MI and entropies per channel, averages NMI.

#### Data Loaders (`create_data_loaders`)
- Batch size 256, workers 4.
- Train: Distortions on.
- Val: Clean images.

#### Training (`train_model`)
- ResNet-18 from scratch (no pretraining).
- AdamW optimizer (lr=0.001, wd=1e-4).
- 90 epochs (standard for ImageNet subsets).
- Soft CE loss.
- Tracks train/val loss/acc with tqdm and TensorBoard.
- Device: CUDA if available.

### Usage
1. Download ImageNet-100 (e.g., from Kaggle or academic sources) and place in `D:\\ImageNet100` with train/val folders and `Labels.json`.
2. Run `python script.py` (replace with your file name).
3. Monitor with `tensorboard --logdir=runs`.

Expected runtime: ~hours on a single GPU for 90 epochs.

### Adaptation for CIFAR-100
The code can be adapted for CIFAR-100 by:
- Changing dataset loading to use `torchvision.datasets.CIFAR100`.
- Updating `num_classes=100`.
- Resizing to 32x32 (or keep 224 with upsampling).
- Adjusting distortions (e.g., smaller patch/block sizes for 32x32 images).
- Example modification:
  ```python
  import torchvision.datasets as datasets

  class DistortedCIFAR100(Dataset):
      # Similar to above, but init with:
      def __init__(self, ...):
          self.dataset = datasets.CIFAR100(root='./data', train=(split=='train'), download=True)
          # Override _load_samples to use self.dataset.data and .targets

  # In loaders: Use transforms.Compose([transforms.ToTensor()]) without resize, or add RandomCrop(32, padding=4), etc.
  ```
TODO: Implement and train on CIFAR-100; preliminary runs suggest similar logic applies, but tune distortion params for smaller images.

## Experiments and Results

Trained on ImageNet-100 with ResNet-18:
- Train Acc: ~XX% (placeholder; run code to get actual).
- Val Acc: ~XX% top-1 (expect 60-70% without pretraining; baselines like standard aug reach ~75%).

For CIFAR-100: TODO - Train and report acc (baselines: ResNet-18 ~70-75% with aug).

### TODO: Comparisons to SOTA and Other Techniques
- Compare to fixed label smoothing (e.g., ε=0.1 uniform).
- Vs. Mixup/CutMix (interpolated labels based on mixing ratios).
- Vs. AutoAugment/RandAugment (policy-based aug without soft labels).
- Robustness tests: Accuracy on noisy/blurry val sets.
- Calibration metrics (ECE) vs. baselines.
- Ablations: NMI vs. proxies (SSIM/PSNR); per-distortion performance.
- SOTA Benchmarks:
  - CIFAR-100: TODO - Search recent papers (e.g., ViTs or EfficientNets hit ~95%; our method's gains on noisy variants).
  - ImageNet-100: TODO - Subset baselines (e.g., ~85% with pretraining; measure relative improvement).
- Hyperparameter sweeps: Bin count in MI, distortion probabilities.
- Scalability: Test on full ImageNet or larger models.

## Installation
- Python 3.8+ with PyTorch 2.0+, torchvision, numpy, tqdm, tensorboard.
- `pip install -r requirements.txt` (create with above).

## License
MIT License.

## Acknowledgments
Built on PyTorch and inspired by information-theoretic DL research.
