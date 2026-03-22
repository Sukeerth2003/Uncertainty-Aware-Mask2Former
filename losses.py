import torch
import torch.nn.functional as F

def relu_evidence(y):
    return F.relu(y)

def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))

def softplus_evidence(y):
    return F.softplus(y)

def kl_divergence(alpha, num_classes, device=None):
    if device is None:
        device = alpha.device
    
    # Heal NaNs and Clamping
    alpha = torch.nan_to_num(alpha, nan=1.0, posinf=100.0, neginf=1.0)
    alpha = torch.clamp(alpha, min=1e-5)
    
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    return first_term + second_term

def loglikelihood_loss(y, alpha, device=None):
    if device is None:
        device = alpha.device
    y = y.float()
    alpha = alpha.float()
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood = torch.sum(
        y * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True
    )
    return -loglikelihood

def evidential_kl_loss(pred_alphas, target_classes, num_classes=None, annealing_step=10):
    """
    Composite Loss with Empty Tensor Check & Label Safety
    """
    # --- FIX 1: HANDLE EMPTY BATCHES ---
    if pred_alphas.numel() == 0:
        return pred_alphas.sum() * 0.0 # Return 0 with gradient attachment
        
    device = pred_alphas.device
    actual_num_classes = pred_alphas.shape[1] 
    
    # --- FIX 2: LABEL CLAMPING ---
    # Ensure targets are within [0, 65]
    safe_targets = target_classes.clamp(0, actual_num_classes - 1)
    
    # Check which ones were valid originally
    valid_mask = (target_classes >= 0) & (target_classes < actual_num_classes)
    
    y_onehot = F.one_hot(safe_targets, actual_num_classes).float()
    
    # Zero out loss for invalid pixels
    y_onehot[~valid_mask] = 0.0

    L_neg = loglikelihood_loss(y_onehot, pred_alphas, device)
    L_kl = kl_divergence(pred_alphas, actual_num_classes, device)
    
    total_loss = torch.mean(L_neg + L_kl)
    return total_loss

def spatial_uncertainty_loss(pred_masks, gt_masks, pred_variance):
    """
    Spatial Uncertainty Loss with Empty Check & Float32
    """
    # --- FIX 1: HANDLE EMPTY BATCHES ---
    if pred_masks.numel() == 0:
        return pred_masks.sum() * 0.0

    # --- FIX 2: NUMERICAL STABILITY ---
    pred_masks = torch.nan_to_num(pred_masks.float(), nan=0.0)
    gt_masks = torch.nan_to_num(gt_masks.float(), nan=0.0)
    pred_variance = torch.nan_to_num(pred_variance.float(), nan=1.0, posinf=100.0)
    
    # Ensure strict positivity
    variance = torch.clamp(pred_variance, min=1e-4, max=1e5)

    pred_probs = torch.sigmoid(pred_masks)
    squared_error = (pred_probs - gt_masks) ** 2
    
    loss = (squared_error / (2 * variance)) + (0.5 * torch.log(variance))
    
    # Check for NaN in final loss
    final_loss = torch.mean(loss)
    if torch.isnan(final_loss):
        return loss.sum() * 0.0 # Fail safe
        
    return final_loss