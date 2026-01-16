"""
Classify images from a local directory using randomized smoothing for certified robustness.
This script combines the Monte Carlo sampling approach from predict.py with local file loading.
"""
import argparse
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from time import time
import datetime
import pandas as pd

from core import Smooth
from architectures import get_architecture
from datasets import get_num_classes, get_normalize_layer

parser = argparse.ArgumentParser(description='Predict on local images using randomized smoothing')
parser.add_argument("image_dir", type=str, help="directory containing images to classify")
parser.add_argument("dataset", type=str, help="dataset name (cifar10 or imagenet) for architecture")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output CSV file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--extensions", type=str, default="jpg,jpeg,png,bmp", help="comma-separated image extensions")
parser.add_argument("--image-size", type=int, default=None, help="resize images to this size (default: 32 for cifar10, 224 for imagenet)")
args = parser.parse_args()

# Class names for different datasets
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

IMAGENET_CLASSES = None  # Will load from file if needed

def get_class_name(prediction, dataset):
    """Convert prediction index to class name."""
    if dataset == 'cifar10':
        if 0 <= prediction < len(CIFAR10_CLASSES):
            return CIFAR10_CLASSES[prediction]
        return f"class_{prediction}"
    elif dataset == 'imagenet':
        # For ImageNet, just return the class index
        return f"class_{prediction}"
    else:
        return f"class_{prediction}"

def get_transform(dataset, image_size=None):
    """Get image preprocessing transform."""
    if image_size is None:
        image_size = 32 if dataset == 'cifar10' else 224
    
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

def load_image(image_path, transform):
    """Load and preprocess a single image."""
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img)
        return img_tensor, None
    except Exception as e:
        return None, str(e)

def main():
    # Determine image size
    image_size = args.image_size
    if image_size is None:
        image_size = 32 if args.dataset == 'cifar10' else 224
    
    print(f"Loading model from: {args.base_classifier}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the base classifier
    checkpoint = torch.load(args.base_classifier, map_location=device)
    
    # Handle different checkpoint formats
    if "arch" in checkpoint:
        # Standard format from train.py
        base_classifier = get_architecture(checkpoint["arch"], args.dataset)
        base_classifier.load_state_dict(checkpoint['state_dict'])
    else:
        # Alternative format - load model without architecture info
        # The state_dict was likely saved without the normalize layer wrapper
        import torch.nn as nn
        from torchvision.models.resnet import BasicBlock
        from datasets import get_normalize_layer
        
        state_dict_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'state_dict'
        state_dict = checkpoint[state_dict_key]
        
        # Create a ResNet18 adapted for CIFAR-10 (3x3 conv1 instead of 7x7)
        class ResNet18_CIFAR(nn.Module):
            def __init__(self, num_classes=10):
                super(ResNet18_CIFAR, self).__init__()
                self.inplanes = 64
                # CIFAR uses 3x3 conv1 with stride 1 and no maxpool
                self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU(inplace=True)
                
                self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
                self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
                self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
                self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(512, num_classes)
                
            def _make_layer(self, block, planes, blocks, stride=1):
                downsample = None
                if stride != 1 or self.inplanes != planes * block.expansion:
                    downsample = nn.Sequential(
                        nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(planes * block.expansion),
                    )
                layers = []
                layers.append(block(self.inplanes, planes, stride, downsample))
                self.inplanes = planes * block.expansion
                for _ in range(1, blocks):
                    layers.append(block(self.inplanes, planes))
                return nn.Sequential(*layers)
            
            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        model = ResNet18_CIFAR(num_classes=10)
        model.load_state_dict(state_dict)
        model = model.to(device)
        
        # Now wrap with normalize layer
        normalize_layer = get_normalize_layer(args.dataset)
        base_classifier = torch.nn.Sequential(normalize_layer, model)
    
    base_classifier = base_classifier.to(device)
    
    # Create the smoothed classifier
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)
    
    # Get image preprocessing transform
    transform = get_transform(args.dataset, image_size)
    
    # Get list of image files
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"ERROR: Image directory does not exist: {args.image_dir}")
        return
    
    extensions = args.extensions.split(',')
    image_files = []
    for ext in extensions:
        image_files.extend(list(image_dir.glob(f"*.{ext.strip()}")))
        image_files.extend(list(image_dir.glob(f"*.{ext.strip().upper()}")))
    
    # Sort for consistent ordering
    image_files = sorted(set(image_files))
    
    if len(image_files) == 0:
        print(f"No images found in {args.image_dir} with extensions: {extensions}")
        return
    
    print(f"Found {len(image_files)} images to classify")
    print(f"Sigma: {args.sigma}, N: {args.N}, Alpha: {args.alpha}")
    print("-" * 80)
    
    # Classify each image
    results = []
    for i, image_path in enumerate(image_files):
        # Load image
        img_tensor, error = load_image(image_path, transform)
        
        if error:
            print(f"[{i+1}/{len(image_files)}] {image_path.name}: ERROR - {error}")
            results.append({
                'filename': image_path.name,
                'path': str(image_path),
                'prediction': -1,
                'class_name': 'ERROR',
                'time': '',
                'error': error
            })
            continue
        
        # Move to device
        img_tensor = img_tensor.to(device)
        
        # Make prediction with randomized smoothing
        before_time = time()
        prediction = smoothed_classifier.predict(img_tensor, args.N, args.alpha, args.batch)
        after_time = time()
        
        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        class_name = get_class_name(prediction, args.dataset)
        
        print(f"[{i+1}/{len(image_files)}] {image_path.name}: {class_name} (class {prediction}) - Time: {time_elapsed}")
        
        results.append({
            'filename': image_path.name,
            'path': str(image_path),
            'prediction': prediction,
            'class_name': class_name,
            'time': time_elapsed,
            'error': ''
        })
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(args.outfile, index=False)
    
    print("-" * 80)
    print(f"✓ Classification complete!")
    print(f"✓ Results saved to: {args.outfile}")
    
    # Print summary
    successful = df[df['prediction'] >= 0]
    if len(successful) > 0:
        print(f"\nSuccessfully classified: {len(successful)}/{len(df)} images")
        print("\nPrediction Summary:")
        prediction_counts = successful['class_name'].value_counts()
        for class_name, count in prediction_counts.items():
            print(f"  {class_name}: {count} images")

if __name__ == "__main__":
    args = parser.parse_args()
    main()
