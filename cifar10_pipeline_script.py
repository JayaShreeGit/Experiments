"""
DiffSmooth CIFAR-10 Pipeline
1. Load a local PNG image file.
2. Perform diffusion-based purification (simplified denoising).
3. Apply local Gaussian smoothing and classify using CIFAR-10 ResNet110.
4. Display the original and purified images with the predicted label.
"""

import os
# Fix OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
from collections import Counter

# Add DiffSmooth directory to path for importing CIFAR-10 architectures
sys.path.append(r'c:\Users\jayas\Documents\PhD\DiffSmooth\DiffSmooth')


def load_cifar10_model(model_path, device='cpu'):
    """Load CIFAR-10 model from the specified path with automatic architecture detection."""
    
    # Check if path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    # Look for checkpoint files
    if os.path.isdir(model_path):
        checkpoint_files = [f for f in os.listdir(model_path) if f.endswith(('.pth', '.tar', '.pt', '.ckpt'))]
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {model_path}")
        checkpoint_path = os.path.join(model_path, checkpoint_files[0])
        print(f"Loading CIFAR-10 model from: {checkpoint_path}")
    else:
        checkpoint_path = model_path
        print(f"Loading CIFAR-10 model from: {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # Get the state dict - try multiple possible keys
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("Using 'model_state_dict' from checkpoint")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("Using 'state_dict' from checkpoint")
    else:
        state_dict = checkpoint
        print("Using checkpoint directly as state_dict")
    
    # Check if state dict has "1." prefix (from DataParallel or custom wrapper)
    first_key = list(state_dict.keys())[0]
    has_prefix = first_key.startswith('1.')
    
    if has_prefix:
        print("Detected model wrapper prefix, removing '1.' from parameter names...")
        # Remove the "1." prefix from all keys
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('1.'):
                new_key = key[2:]  # Remove "1." prefix
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
        print(f"Fixed {len(state_dict)} parameter names")
    
    # Try to determine architecture from checkpoint or state dict structure
    arch_found = False
    model = None
    
    # Check if architecture is stored in checkpoint
    if 'arch' in checkpoint:
        arch = checkpoint['arch']
        print(f"Architecture from checkpoint: {arch}")
    else:
        print("No architecture info in checkpoint, inferring from state dict...")
        # Infer from state dict structure
        # Count number of layer groups to determine architecture
        layer_keys = [k for k in state_dict.keys() if k.startswith('layer')]
        print(f"Found {len(layer_keys)} layer keys")
    
    # Try different architectures in order - Try adapted ResNet18 first, then CIFAR-10 specific architectures
    architectures_to_try = [
        ('ResNet18-CIFAR-Adapted', lambda: create_resnet18_cifar_adapted()),
        ('CIFAR ResNet20', 20),
        ('CIFAR ResNet32', 32),
        ('CIFAR ResNet44', 44),
        ('CIFAR ResNet56', 56),
        ('CIFAR ResNet110', 110),
        ('ResNet18', lambda: models.resnet18(num_classes=10)),
        ('ResNet34', lambda: models.resnet34(num_classes=10)),
        ('ResNet50', lambda: models.resnet50(num_classes=10)),
    ]
    
    for arch_item in architectures_to_try:
        try:
            if isinstance(arch_item, tuple) and len(arch_item) == 2:
                arch_name, arch_config = arch_item
            else:
                continue
                
            print(f"Trying {arch_name}...")
            
            if isinstance(arch_config, int):
                # CIFAR ResNet with specific depth
                from archs.cifar_resnet import resnet
                model = resnet(depth=arch_config, num_classes=10, block_name='BasicBlock')
            else:
                # Standard ResNet
                model = arch_config()
            
            # Try to load the state dict
            model.load_state_dict(state_dict, strict=True)
            print(f"[SUCCESS] Successfully loaded model as {arch_name}")
            arch_found = True
            break
        except Exception as e:
            print(f"[FAILED] {arch_name} failed: {str(e)[:100]}")
            continue
    
    if not arch_found or model is None:
        raise RuntimeError("Could not load model with any known architecture")
    
    model = model.to(device)
    model.eval()
    
    print("CIFAR-10 model loaded successfully!")
    return model


def resnet110_cifar():
    """Helper function to create CIFAR ResNet110."""
    from archs.cifar_resnet import resnet
    return resnet(depth=110, num_classes=10, block_name='BasicBlock')


def create_cifar_resnet(depth):
    """Helper function to create CIFAR ResNet of various depths."""
    from archs.cifar_resnet import resnet
    return resnet(depth=depth, num_classes=10, block_name='BasicBlock')


def create_resnet18_cifar_adapted():
    """Create ResNet18 adapted for CIFAR-10 with 3x3 conv1."""
    import torchvision.models as models
    model = models.resnet18(num_classes=10)
    # Replace the first conv layer from 7x7 to 3x3 for CIFAR-10
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove maxpool for CIFAR-10
    model.maxpool = nn.Identity()
    return model


# -----------------------
# 1. Diffusion purification (simplified one-step denoising)
# -----------------------
class SimpleDiffusionPurifier(nn.Module):
    """Simple one-step diffusion purifier."""
    def __init__(self, beta_t=0.02):
        super().__init__()
        self.beta_t = beta_t
    
    def denoise(self, x_noisy, eps_pred):
        alpha_t = 1 - self.beta_t
        return (x_noisy - (1 - alpha_t)**0.5 * eps_pred) / (alpha_t**0.5)
    
    def forward(self, x):
        noise = torch.randn_like(x)
        x_noisy = x + 0.1 * noise  # Add Gaussian noise
        eps_pred = noise  # Mimic predicted noise (simplified)
        x_denoised = self.denoise(x_noisy, eps_pred)
        return torch.clamp(x_denoised, 0, 1)


# -----------------------
# 2. Local smoothing-based certification (classification)
# -----------------------
class LocalSmoothingCertifier:
    def __init__(self, model, local_noise_sd=0.25, num_samples=5, device='cpu'):
        self.model = model.to(device).eval()
        self.local_noise_sd = local_noise_sd
        self.num_samples = num_samples
        self.device = device
    
    @torch.no_grad()
    def certify(self, x):
        """Apply Gaussian perturbations and aggregate predictions."""
        logits_list = []
        for _ in range(self.num_samples):
            noisy_x = x + torch.randn_like(x) * self.local_noise_sd
            preds = self.model(noisy_x.to(self.device))
            logits_list.append(preds)
        avg_logits = torch.mean(torch.stack(logits_list), dim=0)
        return torch.argmax(avg_logits, dim=1)


# -----------------------
# 3. Full DiffSmooth pipeline for CIFAR-10
# -----------------------
def diffsmooth_cifar10_pipeline(image_path, class_labels=None, show_images=True, run_number=None, classifier=None, device=None):
    """Runs purification + smoothing pipeline on CIFAR-10 images."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure the file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load and preprocess image for CIFAR-10 (32x32)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert("RGB")
    image_tensor = transform(img).unsqueeze(0).to(device)

    # Step 1: Diffusion purification
    purifier = SimpleDiffusionPurifier()
    purified_image = purifier(image_tensor)

    # Step 2: Classification with local smoothing - using CIFAR-10 ResNet110 model
    if classifier is None:
        cifar10_model_path = r"C:\Users\jayas\Documents\PhD\random_smoothing\models\cifar10_test"
        classifier = load_cifar10_model(cifar10_model_path, device)
    certifier = LocalSmoothingCertifier(classifier, local_noise_sd=0.25, num_samples=5, device=device)
    prediction = certifier.certify(purified_image)

    # Fetch predicted class label
    class_idx = prediction.item()
    if class_labels is not None and class_idx < len(class_labels):
        label = class_labels[class_idx]
    else:
        label = f"Class {class_idx}"

    # Print results for this run
    run_info = f"Run {run_number}: " if run_number is not None else ""
    print(f"{run_info}Predicted Class {class_idx}: {label}")

    # Display results (only for first run or if explicitly requested)
    if show_images:
        # Convert tensors to numpy arrays for visualization
        original_np = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        purified_np = purified_image.squeeze().permute(1, 2, 0).detach().cpu().numpy()

        plt.figure(figsize=(7, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(original_np)
        plt.title("Original Image (CIFAR-10)")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(purified_np)
        title = f"Purified Image\nPredicted: {label}"
        if run_number is not None:
            title = f"Run {run_number} - " + title
        plt.title(title)
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    return class_idx, label


def run_multiple_diffsmooth_cifar10(image_path, num_runs=10, class_labels=None, classifier=None, device=None):
    """Run DiffSmooth pipeline multiple times on CIFAR-10 images and collect statistics."""
    print(f"Running DiffSmooth CIFAR-10 pipeline {num_runs} times on: {image_path}")
    print("=" * 60)
    
    predictions = []
    labels = []
    
    # Load classifier once if not provided
    if classifier is None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        cifar10_model_path = r"C:\Users\jayas\Documents\PhD\random_smoothing\models\cifar10_test"
        classifier = load_cifar10_model(cifar10_model_path, device)
    
    for i in range(num_runs):
        # Disable image display for all runs except first
        show_images = False
        class_idx, label = diffsmooth_cifar10_pipeline(image_path, class_labels, show_images, i + 1, classifier=classifier, device=device)
        predictions.append(class_idx)
        labels.append(label)
    
    # Analyze results
    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS:")
    print("=" * 60)
    
    # Count predictions
    prediction_counts = Counter(predictions)
    
    print(f"Total runs: {num_runs}")
    print(f"Unique predictions: {len(prediction_counts)}")
    
    print("\nPrediction frequency:")
    for class_idx, count in prediction_counts.most_common():
        if class_labels and class_idx < len(class_labels):
            class_name = class_labels[class_idx]
        else:
            class_name = f"Class {class_idx}"
        percentage = (count / num_runs) * 100
        print(f"  Class {class_idx} ({class_name}): {count}/{num_runs} times ({percentage:.1f}%)")
    
    # Most frequent prediction
    most_common_class, most_common_count = prediction_counts.most_common(1)[0]
    if class_labels and most_common_class < len(class_labels):
        most_common_name = class_labels[most_common_class]
    else:
        most_common_name = f"Class {most_common_class}"
    
    print(f"\nMost frequent prediction: Class {most_common_class} ({most_common_name})")
    print(f"Confidence: {most_common_count}/{num_runs} ({(most_common_count/num_runs)*100:.1f}%)")
    
    return predictions, labels


def process_all_cifar10_samples(samples_dir, num_runs=5):
    """Process all CIFAR-10 PNG files in the samples directory."""
    
    # CIFAR-10 class labels
    class_labels = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    print(f"Using CIFAR-10 class labels: {class_labels}")

    # Find all PNG files in the directory
    if not os.path.exists(samples_dir):
        print(f"Directory not found: {samples_dir}")
        return
    
    png_files = [f for f in os.listdir(samples_dir) if f.lower().endswith('.png')]
    if not png_files:
        print(f"No PNG files found in {samples_dir}")
        return
    
    # Sort files for consistent processing order
    png_files.sort()
    
    print(f"Found {len(png_files)} PNG files in {samples_dir}")
    print("=" * 80)
    
    # Load model once for all images
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading CIFAR-10 model once for all images...")
    cifar10_model_path = r"C:\Users\jayas\Documents\PhD\random_smoothing\models\cifar10_test"
    classifier = load_cifar10_model(cifar10_model_path, device)
    print(f"Model loaded successfully!\n")
    
    all_results = []
    
    for i, png_file in enumerate(png_files):
        image_path = os.path.join(samples_dir, png_file)
        print(f"\n[{i+1}/{len(png_files)}] Processing: {png_file}")
        print("-" * 60)
        
        try:
            # Extract expected class from filename if available
            expected_class = None
            if '_lbl' in png_file:
                try:
                    # Extract class number from filename like "sample_000_lbl0_airplane..."
                    lbl_part = png_file.split('_lbl')[1].split('_')[0]
                    expected_class = int(lbl_part)
                except:
                    pass
            
            # Run DiffSmooth pipeline
            predictions, pred_labels = run_multiple_diffsmooth_cifar10(
                image_path, 
                num_runs=num_runs, 
                class_labels=class_labels,
                classifier=classifier,
                device=device
            )
            
            # Get most common prediction
            prediction_counts = Counter(predictions)
            most_common_class, most_common_count = prediction_counts.most_common(1)[0]
            confidence = (most_common_count / num_runs) * 100
            
            # Store results
            result = {
                'filename': png_file,
                'expected_class': expected_class,
                'predicted_class': most_common_class,
                'confidence': confidence,
                'all_predictions': predictions,
                'prediction_counts': dict(prediction_counts)
            }
            all_results.append(result)
            
            # Show comparison if expected class is available
            if expected_class is not None:
                match_status = "[CORRECT]" if expected_class == most_common_class else "[INCORRECT]"
                expected_name = class_labels[expected_class] if expected_class < len(class_labels) else f"Class {expected_class}"
                predicted_name = class_labels[most_common_class] if most_common_class < len(class_labels) else f"Class {most_common_class}"
                print(f"Expected: {expected_class} ({expected_name})")
                print(f"Predicted: {most_common_class} ({predicted_name}) - {confidence:.1f}% confidence")
                print(f"Result: {match_status}")
            else:
                predicted_name = class_labels[most_common_class] if most_common_class < len(class_labels) else f"Class {most_common_class}"
                print(f"Predicted: {most_common_class} ({predicted_name}) - {confidence:.1f}% confidence")
            
        except Exception as e:
            print(f"[ERROR] Error processing {png_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY (CIFAR-10)")
    print("=" * 80)
    
    if all_results:
        total_images = len(all_results)
        correct_predictions = 0
        total_confidence = 0
        
        # Calculate accuracy and confidence
        for result in all_results:
            total_confidence += result['confidence']
            if result['expected_class'] is not None and result['expected_class'] == result['predicted_class']:
                correct_predictions += 1
        
        avg_confidence = total_confidence / total_images if total_images > 0 else 0
        
        print(f"Total images processed: {total_images}")
        print(f"Average confidence: {avg_confidence:.1f}%")
        
        # Show accuracy if we have expected classes
        images_with_expected = len([r for r in all_results if r['expected_class'] is not None])
        if images_with_expected > 0:
            accuracy = (correct_predictions / images_with_expected) * 100
            print(f"Accuracy: {correct_predictions}/{images_with_expected} ({accuracy:.1f}%)")
        
        # Show prediction distribution
        all_predicted_classes = [r['predicted_class'] for r in all_results]
        prediction_distribution = Counter(all_predicted_classes)
        
        print(f"\nPredicted class distribution:")
        for class_idx, count in prediction_distribution.most_common():
            class_name = class_labels[class_idx] if class_idx < len(class_labels) else f"Class {class_idx}"
            percentage = (count / total_images) * 100
            print(f"  Class {class_idx} ({class_name}): {count} images ({percentage:.1f}%)")
        
        # Show per-class accuracy breakdown if we have expected classes
        if images_with_expected > 0:
            print(f"\nPer-class accuracy breakdown:")
            expected_classes = [r['expected_class'] for r in all_results if r['expected_class'] is not None]
            unique_expected = set(expected_classes)
            
            for expected_class in sorted(unique_expected):
                class_results = [r for r in all_results if r['expected_class'] == expected_class]
                class_correct = len([r for r in class_results if r['predicted_class'] == expected_class])
                class_total = len(class_results)
                class_accuracy = (class_correct / class_total) * 100 if class_total > 0 else 0
                class_name = class_labels[expected_class] if expected_class < len(class_labels) else f"Class {expected_class}"
                print(f"  Class {expected_class} ({class_name}): {class_correct}/{class_total} ({class_accuracy:.1f}%)")
    
    return all_results


# -----------------------
# 4. Main execution
# -----------------------
if __name__ == "__main__":
    # CIFAR-10 class labels
    cifar10_labels = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    # Path to the samples directory (update this to your actual path)
    samples_directory = r"C:\Users\jayas\Documents\PhD\advdiff\advdiff\samples_png"
    
    # Settings - number of runs per image for confidence estimation
    num_runs_per_image = 5
    
    print("DiffSmooth CIFAR-10 Pipeline - Processing All Images")
    print("=" * 80)
    print(f"Image size: 32x32 (CIFAR-10)")
    print(f"Model: ResNet110 trained on CIFAR-10")
    print(f"Noise level: 0.25")
    print(f"Runs per image: {num_runs_per_image}")
    print("=" * 80)
    
    # Process all PNG files in the directory
    results = process_all_cifar10_samples(samples_directory, num_runs_per_image)
    
    print(f"\n[SUCCESS] Processing complete! Processed {len(results) if results else 0} images.")
