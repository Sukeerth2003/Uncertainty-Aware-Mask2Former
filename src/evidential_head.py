import torch
import torch.nn as nn
import torch.nn.functional as F

class EvidentialClassHead(nn.Module):
    """
    Replaces the standard Linear layer in the Transformer Decoder.
    Outputs Dirichlet parameters (alphas) instead of standard logits.
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        # CRITICAL FIX: Mask2Former requires (Num_Classes + 1) outputs 
        # to account for the "No Object" (Void) class.
        # If num_classes=65, we create 66 outputs.
        self.num_outputs = num_classes + 1
        self.linear = nn.Linear(input_dim, self.num_outputs)
        
    def forward(self, x):
        logits = self.linear(x)
        
        # Evidential Theory Requirement:
        # Output must be non-negative (Evidence >= 0).
        # We use Softplus to ensure this.
        # alpha = evidence + 1 (Ensures alpha >= 1, preventing NaNs in LogGamma)
        evidence = F.softplus(logits)
        alpha = evidence + 1
        
        return alpha