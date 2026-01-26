"""
AdvDiff for CIFAR-10 using Class-Conditional Diffusion Model
This script properly replicates the AdvDiff methodology on CIFAR-10:
1. Generate images of a specific class using conditional diffusion
2. Apply adversarial guidance to make them misclassified
"""

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

# For Hugging Face conditional DDPM model
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from typing import List, Optional, Tuple, Union


def get_cifar10_labels():
    """Get CIFAR-10 class labels"""
    return [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]


def load_your_cifar10_model():
    """Load CIFAR-10 trained victim model"""
    print("Loading victim model...")
    
    from torchvision.models import resnet18
    import torch.nn as nn
    
    model = resnet18(weights=None)
    # Modify for CIFAR-10: 10 classes and smaller input size
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    
    # Load trained weights
    checkpoint = torch.load('C:\\Users\\jayas\\Documents\\PhD\\random_smoothing\\models\\cifar10_test\\checkpoint.pth', 
                           map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def get_target_label(logits, original_label, device):
    """
    Get adversarial target label (second-most-likely class)
    Same strategy as original AdvDiff paper
    """
    rates, indices = logits.sort(1, descending=True) 
    
    tar_label = torch.zeros_like(original_label).to(device)
    
    for i in range(original_label.shape[0]):
        if original_label[i] == indices[i][0]:  # correctly classified
            tar_label[i] = indices[i][1]  # target second-most-likely
        else:
            tar_label[i] = indices[i][0]  # target most-likely
    
    return tar_label


class ConditionalDDPMPipeline(DiffusionPipeline):
    """Custom pipeline for class-conditional CIFAR-10 diffusion"""
    
    def __init__(self, unet, scheduler, num_classes: int, vic_model=None):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        self.num_classes = num_classes
        self.vic_model = vic_model
        self._device = unet.device

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        class_labels: Optional[torch.Tensor] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
    ):
        """Standard generation without adversarial guidance"""
        
        # Ensure class_labels is on the same device
        class_labels = class_labels.to(self._device)
        if class_labels.ndim == 0:
            class_labels = class_labels.unsqueeze(0).expand(batch_size)
        else:
            class_labels = class_labels.expand(batch_size)

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        image = randn_tensor(image_shape, generator=generator, device=self._device)

        # Set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # Predict noise conditioned on class
            model_output = self.unet(image, t, class_labels).sample
            # Denoise step
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        # Normalize to [0, 1]
        image = (image / 2 + 0.5).clamp(0, 1)
        
        if output_type == "pil":
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            image = self.numpy_to_pil(image)
        elif output_type == "numpy":
            image = image.cpu().permute(0, 2, 3, 1).numpy()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

    def adversarial_sample(
        self,
        batch_size: int,
        class_labels: torch.Tensor,
        num_inference_steps: int = 50,
        K: int = 5,
        s: float = 1.0,
        a: float = 0.5,
        early_stop: bool = True,
    ):
        """
        Adversarial sampling following AdvDiff methodology:
        - K iterations of adversarial prior refinement
        - Adversarial guidance during early denoising steps (0-20%)
        - Early stopping when attack succeeds
        """
        
        print(f"\n{'='*60}")
        print(f"AdvDiff Adversarial Sampling")
        print(f"  Class: {class_labels[0].item()}")
        print(f"  Samples: {batch_size}")
        print(f"  Iterations (K): {K}")
        print(f"  Latent guidance (s): {s}")
        print(f"  Prior guidance (a): {a}")
        print(f"{'='*60}\n")
        
        # Ensure class_labels on correct device
        class_labels = class_labels.to(self._device)
        if class_labels.ndim == 0:
            class_labels = class_labels.unsqueeze(0).expand(batch_size)
        
        # Image shape
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        # Initial noise (adversarial prior)
        pri_img = torch.randn(image_shape, device=self._device).requires_grad_(True)
        
        # Outer loop: Adversarial prior refinement
        for k in range(K):
            print(f"\n--- Adversarial Iteration {k+1}/{K} ---")
            
            # Start from current prior
            img = pri_img.detach().requires_grad_(True)
            
            # Set timesteps
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps
            total_steps = len(timesteps)
            
            # Inner loop: DDIM/DDPM denoising with adversarial guidance
            for i, t in enumerate(timesteps):
                index = total_steps - i - 1
                
                # Standard diffusion step
                with torch.no_grad():
                    model_output = self.unet(img, t, class_labels).sample
                    img = self.scheduler.step(model_output, t, img).prev_sample
                
                # Adversarial guidance during early timesteps (0-20%)
                if index > total_steps * 0 and index <= total_steps * 0.2:
                    with torch.enable_grad():
                        # Enable gradients for adversarial update
                        img_adv = img.detach().requires_grad_(True)
                        
                        # Decode to pixel space (for CIFAR-10, already in pixel space)
                        # Normalize to [0, 1] for victim model
                        img_normalized = (img_adv / 2 + 0.5).clamp(0, 1)
                        
                        # Get victim model predictions
                        logits = self.vic_model(img_normalized)
                        log_probs = F.log_softmax(logits, dim=-1)
                        
                        # Get adversarial target (second-most-likely class)
                        tar_label = get_target_label(logits, class_labels, self._device)
                        
                        # Print target information on first guidance step
                        if i == 0 or (index == int(total_steps * 0.2)):
                            pred = torch.argmax(log_probs, dim=1)
                            print(f"  [Adversarial Guidance] Ground truth: {class_labels[0].item()} | Current pred: {pred[0].item()} | Target: {tar_label[0].item()}")
                        
                        # Adversarial loss: maximize log probability of target class
                        selected = log_probs[range(len(logits)), tar_label]
                        
                        # Compute gradient
                        gradient = torch.autograd.grad(selected.sum(), img_adv)[0]
                    
                    # Apply adversarial gradient
                    with torch.no_grad():
                        img = img + s * gradient.float()
                
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"  Step {i+1}/{total_steps}")
            
            # Decode final image
            img_final = (img / 2 + 0.5).clamp(0, 1)
            
            # Evaluate attack success
            with torch.no_grad():
                logits = self.vic_model(img_final)
                log_probs = F.log_softmax(logits, dim=-1)
                pred = torch.argmax(log_probs, dim=1)
                
                # Get final target labels for reference
                tar_label = get_target_label(logits, class_labels, self._device)
                
                # Count successful attacks (misclassified)
                success_mask = (pred != class_labels)
                success_num = success_mask.sum().item()
                
                print(f"\n  Ground truth class: {class_labels[0].item()}")
                print(f"  Adversarial target: {tar_label[0].item()}")
                print(f"  Final predictions: {pred.cpu().numpy()}")
                print(f"  Attack success: {success_num}/{batch_size}")
            
            # Early stopping if all attacks succeed
            if early_stop and success_num == batch_size:
                print(f"\n✓ All attacks successful! Stopping early at iteration {k+1}/{K}")
                return img_final, success_mask
            
            # Update adversarial prior for next iteration
            if k < K - 1:  # Don't compute gradient on last iteration
                with torch.enable_grad():
                    img_for_prior = img.detach().requires_grad_(True)
                    img_normalized = (img_for_prior / 2 + 0.5).clamp(0, 1)
                    
                    logits = self.vic_model(img_normalized)
                    log_probs = F.log_softmax(logits, dim=-1)
                    tar_label = get_target_label(logits, class_labels, self._device)
                    selected = log_probs[range(len(logits)), tar_label]
                    
                    gradient = torch.autograd.grad(selected.sum(), img_for_prior)[0]
                
                # Update prior
                with torch.no_grad():
                    pri_img = pri_img + a * gradient.float()
        
        # Return final images (only successful attacks if using early stop)
        return img_final, success_mask

    def to(self, device: torch.device):
        self._device = device
        self.unet.to(device)
        if self.vic_model is not None:
            self.vic_model.to(device)
        return self


def load_conditional_diffusion_model(vic_model):
    """Load class-conditional CIFAR-10 DDPM model from Hugging Face"""
    print("Loading conditional CIFAR-10 DDPM model from Hugging Face...")
    
    repo_id = "Ketansomewhere/cifar10_conditional_diffusion1"
    num_classes = 10
    
    try:
        unet = UNet2DModel.from_pretrained(repo_id, subfolder="unet").to(device)
        scheduler = DDPMScheduler.from_pretrained(repo_id, subfolder="scheduler")
        
        # Use DDIM scheduler for faster sampling
        scheduler = DDIMScheduler.from_config(scheduler.config)
        
        pipeline = ConditionalDDPMPipeline(
            unet=unet, 
            scheduler=scheduler, 
            num_classes=num_classes,
            vic_model=vic_model
        ).to(device)
        
        print("Class-conditional CIFAR-10 DDPM model loaded successfully")
        return pipeline
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have installed: pip install diffusers transformers")
        raise


parser = argparse.ArgumentParser(description='AdvDiff for CIFAR-10 with Class-Conditional Diffusion')
parser.add_argument('--batch-size', type=int, default=4, help='Number of samples per class')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--num-inference-steps', type=int, default=50, help='Number of diffusion steps (DDIM)')
parser.add_argument('--K', type=int, default=5, help='Number of adversarial prior refinement iterations')
parser.add_argument('--s', type=float, default=1.0, help='Latent adversarial guidance strength')
parser.add_argument('--a', type=float, default=0.5, help='Prior adversarial guidance strength')
parser.add_argument('--save-dir', type=str, default='adversarial_examples_conditional/', help='Output directory')
parser.add_argument('--target-class', type=int, default=None, help='Target CIFAR-10 class (0-9), None for all')
parser.add_argument('--all-classes', action='store_true', help='Generate samples for all 10 CIFAR-10 classes')
parser.add_argument('--early-stop', action='store_true', default=True, help='Stop when attack succeeds')
args = parser.parse_args()


def main():
    print("="*60)
    print("AdvDiff: CIFAR-10 Class-Conditional Adversarial Generation")
    print("="*60)
    
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
    print(f"\nLoaded {len(cifar10_labels)} CIFAR-10 class labels")
    
    # Load victim model
    print("\nLoading victim model...")
    vic_model = load_your_cifar10_model()
    vic_model = vic_model.to(device)
    vic_model.eval()
    print("Victim model loaded successfully")
    
    # Load conditional diffusion model
    pipeline = load_conditional_diffusion_model(vic_model)
    
    # Determine which classes to generate samples for
    if args.all_classes or args.target_class is None:
        target_classes = list(range(10))
        print(f"\nGenerating adversarial samples for all {len(target_classes)} classes")
    else:
        target_classes = [args.target_class]
        print(f"\nGenerating adversarial samples for class {args.target_class}")
    
    # Store all generated samples
    all_images = []
    all_labels = []
    all_predictions = []
    all_class_names = []
    
    # Generate adversarial samples for each class
    for class_idx in target_classes:
        class_name = cifar10_labels[class_idx]
        
        print(f"\n{'='*60}")
        print(f"Class {class_idx}: {class_name}")
        print(f"{'='*60}")
        
        # Prepare class labels
        class_labels = torch.tensor([class_idx] * args.batch_size).to(device)
        
        # Generate adversarial samples using AdvDiff methodology
        adv_images, success_mask = pipeline.adversarial_sample(
            batch_size=args.batch_size,
            class_labels=class_labels,
            num_inference_steps=args.num_inference_steps,
            K=args.K,
            s=args.s,
            a=args.a,
            early_stop=args.early_stop
        )
        
        # Test final results on victim model
        print(f"\n--- Final Evaluation ---")
        with torch.no_grad():
            logits = vic_model(adv_images)
            predicted_classes = torch.argmax(logits, dim=1)
            
            # Get target classes for reference
            class_labels_eval = torch.tensor([class_idx] * args.batch_size).to(device)
            target_classes_eval = get_target_label(logits, class_labels_eval, device)
            
            print(f"Ground truth class: {class_idx} ({class_name})")
            print(f"Adversarial targets: {target_classes_eval.cpu().numpy()}")
            print(f"Final predictions: {predicted_classes.cpu().numpy()}")
            
            # Calculate attack success rate
            success_rate = (predicted_classes != class_idx).float().mean()
            print(f"Attack success rate: {success_rate.item()*100:.2f}%")
        
        # Store results
        labels = torch.full((args.batch_size,), class_idx)
        class_names = [class_name] * args.batch_size
        
        # Convert images to numpy (H,W,C) format
        save_img = adv_images.permute(0, 2, 3, 1).cpu().numpy()
        
        all_images.append(save_img)
        all_labels.append(labels.numpy())
        all_predictions.append(predicted_classes.cpu().numpy())
        all_class_names.extend(class_names)
        
        print(f"\nGenerated {len(adv_images)} adversarial images for class {class_idx} ({class_name})")
    
    # Combine all samples
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    # Save results
    output_file = os.path.join(args.save_dir, 'adversarial_examples_conditional.npz')
    np.savez(output_file, 
             images=all_images,
             labels=all_labels,
             predictions=all_predictions,
             class_names=np.array(all_class_names, dtype=object))
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Saved to: {output_file}")
    print(f"Total samples: {len(all_images)}")
    print(f"Classes: {len(target_classes)}")
    print(f"Samples per class: {args.batch_size}")
    print(f"Adversarial iterations (K): {args.K}")
    print(f"Diffusion steps: {args.num_inference_steps}")
    print(f"Guidance strengths: s={args.s}, a={args.a}")
    
    # Calculate overall attack success rate
    overall_success = (all_predictions != all_labels).mean()
    print(f"\nOverall attack success rate: {overall_success*100:.2f}%")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
