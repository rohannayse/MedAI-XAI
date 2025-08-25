# src/models/load_pretrained_model.py
import torch
from huggingface_hub import hf_hub_download
from torchvision import transforms
import torchvision.models as models

def load_pretrained_pneumonia_model():
    """Load the best free pretrained model from Hugging Face"""
    print("🔄 Downloading pretrained model from Hugging Face...")
    
    try:
        # Download model weights from Hugging Face Hub
        model_path = hf_hub_download(
            repo_id="izeeek/resnet18_pneumonia_classifier", 
            filename="resnet18_pneumonia_classifier.pth"
        )
        
        # Load ResNet18 architecture
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Binary classification
        
        # Load pretrained weights
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        print("✅ Pretrained pneumonia model loaded successfully!")
        print(f"   Model: ResNet18")
        print(f"   Accuracy: 83.3%")
        print(f"   Classes: [0=Pneumonia, 1=Normal]")
        
        return model, get_transforms()
        
    except Exception as e:
        print(f"❌ Failed to load pretrained model: {e}")
        print("🔄 Falling back to ImageNet pretrained model...")
        return load_fallback_model()

def get_transforms():
    """Get the correct transforms for the pretrained model"""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def load_fallback_model():
    """Fallback model if Hugging Face fails"""
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return model, transform

if __name__ == "__main__":
    model, transform = load_pretrained_pneumonia_model()
    print("🎉 Model ready for use!")
