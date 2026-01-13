import numpy as np
from PIL import Image
import os

def get_cifar10_labels():
    """Get CIFAR-10 class labels."""
    return [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

# Choose which NPZ file to load from
# Option 1: Original advdiff.py output (single class)
# NPZ = os.path.join('advdiff', 'adversarial_examples.npz')

# Option 2: Custom model output (multiple classes)
NPZ = os.path.join('adversarial_examples_mc', 'adversarial_examples_mc.npz')

# Check if custom model file exists, fallback to original if not
if not os.path.exists(NPZ):
    print(f"Custom model output not found at {NPZ}")
    print("Falling back to original advdiff.py output...")
    NPZ = os.path.join('advdiff', 'adversarial_examples.npz')
    if not os.path.exists(NPZ):
        print(f"No adversarial examples found! Please run advdiff_custom_model.py first.")
        exit(1)

print(f"Loading adversarial examples from: {NPZ}")
OUT = os.path.join('samples_png')
os.makedirs(OUT, exist_ok=True)

z = np.load(NPZ, allow_pickle=True)
print(f"Available arrays in NPZ file: {z.files}")

# Load data based on new format
if 'images' in z.files:
    imgs = z['images']   # shape (N,H,W,C)
    labels = z['labels'] if 'labels' in z.files else None
    class_names = z['class_names'] if 'class_names' in z.files else None
else:
    # Fallback to old format
    imgs = z[z.files[0]]   # shape (N,H,W,C)
    labels = z[z.files[1]] if len(z.files) > 1 else None
    class_names = None

print(f"Images shape: {imgs.shape}")
if labels is not None:
    print(f"Labels shape: {labels.shape}")
if class_names is not None:
    print(f"Class names shape: {class_names.shape}")

# Load CIFAR-10 labels
cifar10_labels = get_cifar10_labels()
print(f"Loaded {len(cifar10_labels)} CIFAR-10 class labels")

# convert floats in [0,1] to uint8
def to_pil(img):
    arr = (img * 255.0).round().astype('uint8')
    if arr.shape[2] == 1:
        arr = arr[:,:,0]
    return Image.fromarray(arr)

def clean_class_name_for_filename(class_name):
    """Clean class name to make it suitable for use in filenames."""
    if class_name is None:
        return "unknown_class"
    
    # Convert to string and clean up
    clean_name = str(class_name)
    
    # Remove or replace problematic characters
    replacements = {
        '/': '_',
        '\\': '_', 
        ':': '_',
        '*': '_',
        '?': '_',
        '"': '_',
        '<': '_',
        '>': '_',
        '|': '_',
        ' ': '_',
        ',': '_',
        "'": '',
        '.': '_',
        '(': '_',
        ')': '_',
        '-': '_'
    }
    
    for old, new in replacements.items():
        clean_name = clean_name.replace(old, new)
    
    # Remove multiple consecutive underscores
    while '__' in clean_name:
        clean_name = clean_name.replace('__', '_')
    
    # Remove leading/trailing underscores
    clean_name = clean_name.strip('_')
    
    # Limit length to avoid very long filenames
    clean_name = clean_name[:40]
    
    return clean_name if clean_name else "unknown_class"

# Save all images
N = imgs.shape[0]
print(f"\nSaving {N} images...")

for i in range(N):
    img = imgs[i]
    pil = to_pil(img)
    
    # Build filename with class information
    fname = f'sample_{i:03d}'
    
    # Add label number
    if labels is not None:
        label_idx = int(labels[i])
        fname += f'_lbl{label_idx}'
        
        # Add CIFAR-10 class name
        if 0 <= label_idx < len(cifar10_labels):
            class_name = cifar10_labels[label_idx]
            clean_name = clean_class_name_for_filename(class_name)
            fname += f'_{clean_name}'
        else:
            fname += f'_class{label_idx}'
    
    # If class names are stored in the file, prefer those
    if class_names is not None and i < len(class_names):
        stored_name = clean_class_name_for_filename(class_names[i])
        # Only add if different from label-based name
        if class_names is None or stored_name not in fname:
            fname += f'_{stored_name}'
    
    fname += '.png'
    pil.save(os.path.join(OUT, fname))
    
    # Print info for each saved image
    info = f"Image {i:03d} -> {fname}"
    if labels is not None:
        label_idx = int(labels[i])
        info += f" - Label: {label_idx}"
        
        # Show class name
        if 0 <= label_idx < len(cifar10_labels):
            info += f" - Class: {cifar10_labels[label_idx]}"
        elif class_names is not None and i < len(class_names):
            info += f" - Class: {class_names[i]}"
    print(info)

print(f'\nWrote {N} images to {OUT}')

# Print summary of unique classes
if labels is not None:
    unique_labels = np.unique(labels)
    print(f"\nGenerated samples for {len(unique_labels)} unique class(es):")
    for label in unique_labels:
        count = np.sum(labels == label)
        label_idx = int(label)
        
        # Get class name
        if 0 <= label_idx < len(cifar10_labels):
            class_name = cifar10_labels[label_idx]
        elif class_names is not None:
            # Find first occurrence of this label to get class name
            first_idx = np.where(labels == label)[0][0]
            if first_idx < len(class_names):
                class_name = class_names[first_idx]
            else:
                class_name = "Unknown"
        else:
            class_name = "Unknown"
            
        print(f"  Class {label_idx}: {class_name} ({count} samples)")
