"""
Classify images from a local directory using a trained CIFAR-10 ResNet18 model.
"""
import torch
import torch.nn as nn
from archs.cifar_resnet import BasicBlock
from datasets import get_normalize_layer
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import pandas as pd

# ============================================================================
# CONFIGURE THESE PATHS
# ============================================================================
IMAGE_DIR = r"C:\Users\jayas\Documents\PhD\advdiff\advdiff\samples_png"
MODEL_PATH = r"C:\Users\jayas\Documents\PhD\random_smoothing\models\cifar10_test\checkpoint.pth"
OUTPUT_FILE = "my_predictions.csv"
IMAGE_EXTENSIONS = "jpg,jpeg,png,bmp"  # comma-separated
# ============================================================================

# CIFAR-10 class names
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

class ResNet18_CIFAR(nn.Module):
    """ResNet18 adapted for CIFAR-10 (32x32 images)"""
    def __init__(self, num_classes=10):
        super(ResNet18_CIFAR, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
    
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
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def load_model(model_path):
    """Load trained ResNet18 CIFAR-10 model from checkpoint."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load checkpoint with proper device mapping
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(str(model_path), map_location=device)
    
    # Get state dict
    state_dict_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'state_dict'
    state_dict = checkpoint[state_dict_key]
    
    # Build ResNet18 model with normalization layer
    base_model = ResNet18_CIFAR(num_classes=10)
    normalize_layer = get_normalize_layer('cifar10')
    model = nn.Sequential(normalize_layer, base_model)
    
    # Handle state dict format - add "1." prefix if needed for Sequential wrapper
    first_key = next(iter(state_dict.keys()))
    if not first_key.startswith('1.'):
        new_state_dict = {}
        for key, value in state_dict.items():
            new_state_dict[f'1.{key}'] = value
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model

def get_transform():
    """Get image preprocessing transform for CIFAR-10."""
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def classify_image(model, image_path, transform, device):
    """Classify a single image."""
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            prediction = output.argmax(dim=1).item()
            confidence = probabilities[prediction].item()
        
        return prediction, confidence, None
    except Exception as e:
        return None, None, str(e)

def main():
    print(f"Loading model from: {MODEL_PATH}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = load_model(MODEL_PATH)
    transform = get_transform()
    
    # Get list of image files
    image_dir = Path(IMAGE_DIR)
    if not image_dir.exists():
        print(f"ERROR: Image directory does not exist: {IMAGE_DIR}")
        return
    
    extensions = IMAGE_EXTENSIONS.split(',')
    image_files = []
    for ext in extensions:
        image_files.extend(list(image_dir.glob(f"*.{ext}")))
        image_files.extend(list(image_dir.glob(f"*.{ext.upper()}")))
    
    if len(image_files) == 0:
        print(f"No images found in {IMAGE_DIR} with extensions: {extensions}")
        return
    
    print(f"Found {len(image_files)} images to classify")
    print("-" * 80)
    
    # Classify each image
    results = []
    for i, image_path in enumerate(image_files, 1):
        prediction, confidence, error = classify_image(model, image_path, transform, device)
        
        if error:
            print(f"[{i}/{len(image_files)}] {image_path.name}: ERROR - {error}")
            results.append({
                'filename': image_path.name,
                'path': str(image_path),
                'prediction': 'ERROR',
                'class_name': 'ERROR',
                'confidence': 0.0,
                'error': error
            })
        else:
            class_name = CIFAR10_CLASSES[prediction]
            print(f"[{i}/{len(image_files)}] {image_path.name}: {class_name} (confidence: {confidence:.4f})")
            results.append({
                'filename': image_path.name,
                'path': str(image_path),
                'prediction': prediction,
                'class_name': class_name,
                'confidence': confidence,
                'error': ''
            })
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print("-" * 80)
    print(f"✓ Classification complete!")
    print(f"✓ Results saved to: {OUTPUT_FILE}")
    
    # Print summary
    if len(df[df['prediction'] != 'ERROR']) > 0:
        print("\nPrediction Summary:")
        for class_name in CIFAR10_CLASSES:
            count = len(df[df['class_name'] == class_name])
            if count > 0:
                print(f"  {class_name}: {count} images")

if __name__ == "__main__":
    main()
