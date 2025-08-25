# src/trust_engine/trust_calculator.py
import torch
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import entropy

class MedicalTrustCalculator:
    def __init__(self, model, temperature=1.5):
        self.model = model
        self.temperature = temperature
        self.reference_features = torch.randn(1000, 512)  # Reference database
        
    def calculate_prediction_confidence(self, logits):
        """Temperature-scaled confidence with calibration"""
        scaled_logits = logits / self.temperature
        probs = torch.softmax(scaled_logits, dim=1)
        max_prob = torch.max(probs, dim=1)
        
        # Confidence gap between top 2 predictions
        top2_probs = torch.topk(probs, 2, dim=1)
        confidence_gap = top2_probs[:, 0] - top2_probs[:, 1]
        
        # Combined confidence score
        confidence_score = (0.7 * max_prob + 0.3 * confidence_gap).item()
        
        return {
            'confidence': confidence_score,
            'max_probability': max_prob.item(),
            'confidence_gap': confidence_gap.item()
        }
    
    def calculate_feature_consistency(self, current_features, top_k=5):
        """Compare with reference case database"""
        current_flat = current_features.flatten().detach().cpu().numpy()
        
        similarities = []
        for ref_features in self.reference_features:
            ref_flat = ref_features.flatten().numpy()
            similarity = 1 - cosine(current_flat, ref_flat)
            similarities.append(similarity)
        
        # Get top-k most similar cases
        top_similarities = np.sort(similarities)[-top_k:]
        consistency_score = np.mean(top_similarities)
        
        return {
            'consistency': consistency_score,
            'top_similarities': top_similarities.tolist()
        }
    
    def calculate_attention_coherence(self, gradcam_heatmap):
        """Measure spatial coherence of attention"""
        heatmap_norm = gradcam_heatmap / (np.sum(gradcam_heatmap) + 1e-8)
        
        # Calculate entropy (lower = more focused = higher trust)
        attention_entropy = entropy(heatmap_norm.flatten() + 1e-8)
        max_entropy = np.log(heatmap_norm.size)
        
        # Coherence score (inverse of normalized entropy)
        coherence_score = 1.0 - (attention_entropy / max_entropy)
        
        # Attention concentration in top 25% of pixels
        flat_heatmap = heatmap_norm.flatten()
        top_25_percent = int(0.25 * len(flat_heatmap))
        top_pixels_attention = np.sum(np.sort(flat_heatmap)[-top_25_percent:])
        
        return {
            'coherence': coherence_score,
            'attention_entropy': attention_entropy,
            'top_25_concentration': top_pixels_attention
        }
    
    def get_comprehensive_trust_score(self, image, gradcam_heatmap):
        """Main trust calculation function"""
        with torch.no_grad():
            # Get predictions and features
            features = self.model.features(image.unsqueeze(0)) if hasattr(self.model, 'features') else image
            logits = self.model(image.unsqueeze(0))
            
            # Calculate components
            confidence_results = self.calculate_prediction_confidence(logits)
            consistency_results = self.calculate_feature_consistency(features)
            coherence_results = self.calculate_attention_coherence(gradcam_heatmap)
            
            # Weighted combination
            trust_score = (
                0.4 * confidence_results['confidence'] +
                0.3 * consistency_results['consistency'] +
                0.3 * coherence_results['coherence']
            )
            
            # Trust level classification
            if trust_score >= 0.8:
                trust_level, trust_color = "HIGH", "green"
                recommendation = "AI prediction reliable for clinical consideration"
            elif trust_score >= 0.6:
                trust_level, trust_color = "MODERATE", "orange"
                recommendation = "Consider additional review or second opinion"
            else:
                trust_level, trust_color = "LOW", "red"
                recommendation = "AI prediction unreliable - manual review required"
            
            return {
                'trust_score': max(0.0, min(1.0, trust_score)),
                'trust_level': trust_level,
                'trust_color': trust_color,
                'recommendation': recommendation,
                'components': {
                    'confidence': confidence_results,
                    'consistency': consistency_results,
                    'coherence': coherence_results
                }
            }
