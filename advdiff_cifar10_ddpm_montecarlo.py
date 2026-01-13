import sys
import os 
import random
import argparse

import torch 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.backends import cudnn
import numpy as np 
from PIL import Image
from torchvision.utils import make_grid
import torch.nn.functional as F

# For Hugging Face DDPM model
from diffusers import DDPMPipeline, DDIMScheduler

def get_cifar10_labels():
    """Get CIFAR-10 class labels"""
    return [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

def load_your_cifar10_model():
    """
    Load your CIFAR-10 trained model.
    Replace this with your actual model loading code.
    """
    print("WARNING: You need to implement loading your CIFAR-10 model!")
    print("This is a placeholder - loading a random ResNet18 for CIFAR-10")
    
    from torchvision.models import resnet18
    import torch.nn as nn
    
    model = resnet18(weights=None)
    # Modify for CIFAR-10: 10 classes and smaller input size
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    
    # TODO: Load your trained weights here
    checkpoint = torch.load('C:\\Users\\jayas\\Documents\\PhD\\random_smoothing\\models\\cifar10_test\\checkpoint.pth', 
                           map_location=device)
    # Extract the model state dict from the checkpoint
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=4, help='Number of samples per class')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num-inference-steps', type=int, default=50, help='Number of diffusion steps')
parser.add_argument('--guidance-scale', type=float, default=1.0, help='Guidance scale (1.0 = no guidance)')
parser.add_argument('--attack-strength', type=float, default=0.1, help='Adversarial attack strength')
parser.add_argument('--attack-steps', type=int, default=10, help='Number of attack optimization steps')
parser.add_argument('--monte-carlo-samples', type=int, default=5, help='Number of Monte Carlo samples to generate and select from')
parser.add_argument('--save-dir', type=str, default='adversarial_examples_mc/')
parser.add_argument('--target-class', type=int, default=None, help='Target CIFAR-10 class to attack (0-9), None for all classes')
parser.add_argument('--model-path', type=str, default='', help='Path to your CIFAR-10 model checkpoint')
parser.add_argument('--all-classes', action='store_true', help='Generate samples for all 10 CIFAR-10 classes')
args = parser.parse_args()

def load_diffusion_model():
    """Load CIFAR-10 DDPM model from Hugging Face"""
    print("Loading google/ddpm-cifar10-32 from Hugging Face...")
    try:
        pipeline = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
        pipeline = pipeline.to(device)
        
        # Use DDIM scheduler for faster sampling
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        
        print("CIFAR-10 DDPM model loaded successfully")
        return pipeline
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have installed: pip install diffusers transformers")
        raise

def adversarial_diffusion_sampling_monte_carlo(pipeline, vic_model, target_class, n_samples, 
                                               num_inference_steps, attack_strength, attack_steps,
                                               n_monte_carlo_samples):
    """
    Generate adversarial samples using Monte Carlo method with diffusion model.
    
    Monte Carlo Approach:
    1. Generate multiple candidate samples from different noise initializations
    2. Evaluate each candidate on the victim model
    3. Select the best adversarial samples (highest attack success)
    
    This provides better exploration of the adversarial space compared to single-sample generation.
    """
    
    print(f"Monte Carlo sampling: Generating {n_monte_carlo_samples} candidates per sample...")
    print(f"Total candidates to generate: {n_samples * n_monte_carlo_samples}")
    
    all_candidates = []
    all_losses = []
    
    # Prepare target labels for victim model
    target_labels = torch.full((1,), target_class, device=device, dtype=torch.long)
    
    # Generate multiple candidates using Monte Carlo sampling
    for mc_idx in range(n_monte_carlo_samples):
        print(f"\n--- Monte Carlo Trial {mc_idx + 1}/{n_monte_carlo_samples} ---")
        
        # Set the number of inference steps
        pipeline.scheduler.set_timesteps(num_inference_steps)
        
        # Start with random noise (different for each MC trial)
        image_shape = (n_samples, 3, 32, 32)
        image = torch.randn(image_shape, device=device)
        
        # Denoise step by step
        for i, t in enumerate(pipeline.scheduler.timesteps):
            # Predict noise residual (standard diffusion step)
            with torch.no_grad():
                noise_pred = pipeline.unet(image, t).sample
            
            # Standard diffusion update
            image = pipeline.scheduler.step(noise_pred, t, image).prev_sample
            
            # Every few steps, add adversarial gradient
            if i % max(1, num_inference_steps // attack_steps) == 0:
                # Enable gradients computation
                vic_model.eval()  # Keep in eval mode but allow gradients
                
                # Enable gradients for this image
                image_for_victim = image.detach().requires_grad_(True)
                
                # Resize to match victim model input (if needed)
                resized_image = F.interpolate(image_for_victim, size=(32, 32), mode='bilinear', align_corners=False)
                
                # Normalize for victim model
                normalized_image = (resized_image + 1.0) / 2.0  # Convert from [-1, 1] to [0, 1]
                
                # Get victim model predictions with gradients enabled
                with torch.enable_grad():
                    logits = vic_model(normalized_image)
                    
                    # Loss: untargeted attack - minimize correct class probability
                    target_labels_batch = target_labels.repeat(n_samples)
                    loss = -F.cross_entropy(logits, target_labels_batch)
                    
                    # Get gradient
                    loss.backward()
                
                # Update image with adversarial gradient
                with torch.no_grad():
                    grad_sign = image_for_victim.grad.sign() if image_for_victim.grad is not None else 0
                    image = image + attack_strength * grad_sign
                    # Clamp to valid range
                    image = torch.clamp(image, -1.0, 1.0)
            
            if (i + 1) % 10 == 0:
                print(f"  Step {i+1}/{num_inference_steps}")
        
        # Final output
        image = (image + 1.0) / 2.0  # Convert from [-1, 1] to [0, 1]
        image = torch.clamp(image, 0.0, 1.0)
        
        # Evaluate adversarial effectiveness for each candidate
        with torch.no_grad():
            logits = vic_model(image)
            
            # Calculate loss for each sample (higher loss = more adversarial)
            target_labels_batch = target_labels.repeat(n_samples)
            sample_losses = F.cross_entropy(logits, target_labels_batch, reduction='none')
            
            # Store candidates and their losses
            for sample_idx in range(n_samples):
                all_candidates.append(image[sample_idx].cpu())
                all_losses.append(sample_losses[sample_idx].item())
        
        print(f"  Avg loss for this trial: {sample_losses.mean().item():.4f}")
    
    # Select best candidates (highest loss = most adversarial)
    print(f"\n--- Monte Carlo Selection ---")
    print(f"Total candidates generated: {len(all_candidates)}")
    
    all_losses = torch.tensor(all_losses)
    all_candidates = torch.stack(all_candidates)
    
    # Reshape to (n_samples, n_monte_carlo_samples, C, H, W)
    all_candidates = all_candidates.view(n_monte_carlo_samples, n_samples, 3, 32, 32).permute(1, 0, 2, 3, 4)
    all_losses = all_losses.view(n_monte_carlo_samples, n_samples).T
    
    # Select best candidate for each sample (highest loss)
    best_indices = torch.argmax(all_losses, dim=1)
    selected_images = []
    
    for sample_idx in range(n_samples):
        best_mc_idx = best_indices[sample_idx].item()
        selected_images.append(all_candidates[sample_idx, best_mc_idx])
        print(f"Sample {sample_idx}: Selected MC trial {best_mc_idx + 1}/{n_monte_carlo_samples} (loss: {all_losses[sample_idx, best_mc_idx].item():.4f})")
    
    selected_images = torch.stack(selected_images).to(device)
    
    return selected_images

def main():
    print("Starting AdvDiff with CIFAR-10 DDPM model + Monte Carlo sampling...")
    
    # Set seeds for reproducibility
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load CIFAR-10 class labels
    cifar10_labels = get_cifar10_labels()
    print(f"Loaded {len(cifar10_labels)} CIFAR-10 class labels")
    
    # Load diffusion model
    diffusion_pipeline = load_diffusion_model()
    
    # Load victim model
    print("\nLoading victim model...")
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading from: {args.model_path}")
        # TODO: Implement your model loading
        vic_model = load_your_cifar10_model()
    else:
        print("No model path provided. Using placeholder model.")
        vic_model = load_your_cifar10_model()
    
    vic_model = vic_model.to(device)
    vic_model.eval()
    print("Victim model loaded successfully")
    
    # Determine which classes to generate samples for
    if args.all_classes or args.target_class is None:
        target_classes = list(range(10))  # All CIFAR-10 classes
        print(f"\nGenerating samples for all {len(target_classes)} classes")
    else:
        target_classes = [args.target_class]
        print(f"\nGenerating samples for class {args.target_class}")
    
    # Store all generated samples
    all_images = []
    all_labels = []
    all_class_names = []
    
    # Generate samples for each target class
    for target_class in target_classes:
        class_name = cifar10_labels[target_class]
        
        print(f"\n{'='*60}")
        print(f"Generating adversarial examples for class {target_class}: {class_name}")
        print(f"  Samples: {args.batch_size}")
        print(f"  Monte Carlo trials: {args.monte_carlo_samples}")
        print(f"  Diffusion steps: {args.num_inference_steps}")
        print(f"  Attack strength: {args.attack_strength}")
        print(f"{'='*60}")
        
        # Use Monte Carlo sampling for better adversarial sample generation
        images = adversarial_diffusion_sampling_monte_carlo(
            diffusion_pipeline,
            vic_model,
            target_class,
            args.batch_size,
            args.num_inference_steps,
            args.attack_strength,
            args.attack_steps,
            args.monte_carlo_samples
        )
        
        # Test on victim model
        print(f"\nTesting adversarial examples on victim model...")
        with torch.no_grad():
            predictions = vic_model(images)
            predicted_classes = torch.argmax(predictions, dim=1)
            
            print(f"Target class: {target_class} ({class_name})")
            print(f"Predicted classes: {predicted_classes.cpu().numpy()}")
            
            # Calculate attack success rate (predictions != target)
            success = (predicted_classes != target_class).float().mean()
            print(f"Attack success rate: {success.item()*100:.2f}%")
        
        # Store results for this class
        labels = torch.full((args.batch_size,), target_class)
        class_names = [class_name] * args.batch_size
        
        # Convert images to numpy (H,W,C) format
        save_img = images.permute(0, 2, 3, 1).cpu().numpy()
        
        all_images.append(save_img)
        all_labels.append(labels.numpy())
        all_class_names.extend(class_names)
        
        print(f"Generated {len(images)} images for class {target_class} ({class_name})")
    
    # Combine all samples
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Save combined results
    output_file = os.path.join(args.save_dir, 'adversarial_examples_mc.npz')
    np.savez(output_file, 
             images=all_images,
             labels=all_labels,
             class_names=np.array(all_class_names, dtype=object))
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: Saved {len(all_images)} adversarial examples to:")
    print(f"  {output_file}")
    print(f"Generated {args.batch_size} samples for each of {len(target_classes)} classes")
    print(f"Used {args.monte_carlo_samples} Monte Carlo trials per sample")
    print(f"Total samples: {len(all_images)}")
    print(f"{'='*60}")

if __name__ == '__main__':
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
