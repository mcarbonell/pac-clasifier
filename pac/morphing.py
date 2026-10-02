"""
Spatial Morphing Utilities for PAC Archetypes.
Allows affine transformation (translation, rotation, scaling) to evaluate invariant matching.
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam


def morph_archetype(archetype, tx, ty, theta, scale, image_shape=(28, 28)):
    """
    Applies an affine transformation to a flattened archetype vector.
    
    archetype: (D,) tensor, where D = H * W
    tx, ty: translation offsets (-1.0 to 1.0)
    theta: rotation angle in radians
    scale: zoom factor (> 0)
    image_shape: tuple of (H, W)
    """
    h, w = image_shape
    if archetype.numel() != h * w:
        raise ValueError(f"Archetype size {archetype.numel()} does not match image_shape {h}x{w} = {h*w}")
        
    img = archetype.view(1, 1, h, w)
    
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    
    # Affine matrix [2, 3]:
    # [ scale*cos(theta), -scale*sin(theta), tx ]
    # [ scale*sin(theta),  scale*cos(theta), ty ]
    matrix = torch.stack([
        torch.stack([scale * cos_t, -scale * sin_t, tx]),
        torch.stack([scale * sin_t,  scale * cos_t, ty])
    ]).unsqueeze(0)  # Shape (1, 2, 3)
    
    grid = F.affine_grid(matrix, [1, 1, h, w], align_corners=False)
    morphed = F.grid_sample(img, grid, align_corners=False)
    
    return morphed.view(-1)


def optimize_morph_fit(target_img, archetype, image_shape=(28, 28), iterations=10, lr=0.01, device=None):
    """
    Optimizes affine parameters (tx, ty, theta, scale) to fit an archetype to a target image.
    Returns the best cosine similarity achieved.
    """
    dev = device if device is not None else target_img.device
    
    tx = torch.tensor(0.0, device=dev, requires_grad=True)
    ty = torch.tensor(0.0, device=dev, requires_grad=True)
    theta = torch.tensor(0.0, device=dev, requires_grad=True)
    scale = torch.tensor(1.0, device=dev, requires_grad=True)
    
    params = [tx, ty, theta, scale]
    optimizer = Adam(params, lr=lr)
    
    best_sim = -1.0
    
    for _ in range(iterations):
        optimizer.zero_grad()
        morphed = morph_archetype(archetype, tx, ty, theta, scale, image_shape=image_shape)
        
        # MSE loss for stable gradient optimization of spatial alignment
        loss = F.mse_loss(morphed, target_img)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            # Evaluate using cosine similarity to stay consistent with PAC
            sim = F.cosine_similarity(target_img.unsqueeze(0), morphed.unsqueeze(0)).item()
            if sim > best_sim:
                best_sim = sim
                
    return best_sim
