import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialUncertaintyHead(nn.Module):
    """
    Predicts a pixel-wise variance map (spatial uncertainty) from the mask features.
    This captures aleatoric uncertainty (e.g., sensor noise, fog, rain).
    """
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        # Simple CNN to map feature channels -> 1 channel (Variance)
        self.layers = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1) # Output 1 channel: Variance
        )
        
    def forward(self, mask_features):
        # mask_features shape: [Batch, Channels, Height, Width]
        variance_logits = self.layers(mask_features)
        
        # Variance must be positive. We use Softplus to strictly enforce > 0.
        # Adding epsilon (1e-6) prevents numerical instability in Log(Variance).
        variance_map = F.softplus(variance_logits) + 1e-6
        
        return variance_map