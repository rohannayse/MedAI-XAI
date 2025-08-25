# src/visualization/medical_gradcam.py
import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class MedicalGradCAM:
    def __init__(self, model, target_layers=None):
        self.model = model
        if target_layers is None:
            target_layers = self._auto_detect_layers()
        self.cam = GradCAM(model=model, target_layers=target_layers)
        
        # Anatomical regions for chest X-rays
        self.anatomical_regions = {
            'left_lung': {'x': 0.1, 'y': 0.2, 'w': 0.35, 'h': 0.6},
            'right_lung': {'x': 0.55, 'y': 0.2, 'w': 0.35, 'h': 0.6},
            'heart': {'x': 0.35, 'y': 0.3, 'w': 0.3, 'h': 0.4},
            'upper_chest': {'x': 0.2, 'y': 0.1, 'w': 0.6, 'h': 0.3},
            'lower_chest': {'x': 0.2, 'y': 0.6, 'w': 0.6, 'h': 0.3}
        }
    
    def _auto_detect_layers(self):
        """Auto-detect appropriate target layers"""
        if hasattr(self.model, 'features'):
            return [self.model.features[-1]]
        elif hasattr(self.model, 'layer4'):
            return [self.model.layer4[-1]]
        else:
            # Find last convolutional layer
            for name, module in reversed(list(self.model.named_modules())):
                if isinstance(module, torch.nn.Conv2d):
                    return [module]
        return [list(self.model.children())[-1]]
    
    def generate_explanation(self, input_tensor, target_class=None):
        """Generate comprehensive medical explanation"""
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Generate GradCAM heatmap
        targets = [ClassifierOutputTarget(target_class)] if target_class else None
        
        try:
            grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
            heatmap = grayscale_cam[0, :]
        except Exception as e:
            print(f"GradCAM failed: {e}, using gradient fallback")
            heatmap = self._gradient_fallback(input_tensor, target_class)
        
        # Medical region analysis
        medical_analysis = self._analyze_medical_regions(heatmap)
        
        # Create visualization
        visualization = self._create_medical_visualization(input_tensor, heatmap)
        
        return {
            'heatmap': heatmap,
            'visualization': visualization,
            'medical_analysis': medical_analysis,
            'peak_attention_coords': self._find_peak_attention(heatmap)
        }
    
    def _analyze_medical_regions(self, heatmap):
        """Analyze attention across anatomical regions"""
        h, w = heatmap.shape
        region_analysis = {}
        
        for region_name, coords in self.anatomical_regions.items():
            # Convert relative to absolute coordinates
            x_start, y_start = int(coords['x'] * w), int(coords['y'] * h)
            x_end, y_end = int((coords['x'] + coords['w']) * w), int((coords['y'] + coords['h']) * h)
            
            region_heatmap = heatmap[y_start:y_end, x_start:x_end]
            
            region_analysis[region_name] = {
                'mean_attention': float(np.mean(region_heatmap)),
                'max_attention': float(np.max(region_heatmap)),
                'attention_percentage': float(np.sum(region_heatmap) / np.sum(heatmap) * 100)
            }
        
        return region_analysis
    
    def _create_medical_visualization(self, original_image, heatmap):
        """Create medical visualization with overlays"""
        if isinstance(original_image, torch.Tensor):
            original_image = original_image.permute(1, 2, 0).cpu().numpy()
        
        # Normalize and ensure RGB
        original_image = np.clip(original_image, 0, 1)
        if len(original_image.shape) == 2:
            original_image = np.stack([original_image] * 3, axis=-1)
        
        # Resize heatmap to match image
        if heatmap.shape != original_image.shape[:2]:
            heatmap = cv2.resize(heatmap, (original_image.shape[15], original_image.shape))
        
        return show_cam_on_image(original_image, heatmap, use_rgb=True)
    
    def _find_peak_attention(self, heatmap):
        """Find peak attention coordinates"""
        peak_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        return {'y': int(peak_idx), 'x': int(peak_idx[15])}
    
    def _gradient_fallback(self, input_tensor, target_class):
        """Fallback attention using gradients"""
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        target_output = output[0, target_class]
        target_output.backward()
        
        gradients = input_tensor.grad.data
        attention = torch.mean(torch.abs(gradients), dim=1).squeeze().cpu().numpy()
        
        # Normalize
        attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
        return attention
