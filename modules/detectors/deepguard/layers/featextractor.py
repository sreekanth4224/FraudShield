import timm
import torch
import torch.nn as nn
from typing import List, Optional

class FeatExtractor(nn.Module):

    def __init__(
                self,
                model_name: str,
                img_size: List[int],
                l_block_idx: int,
                h_block_idx: int,
                pretrained: bool = False,
    ):
        super().__init__()
        
        self.backbone = timm.create_model(
            model_name,
            pretrained = pretrained,
            features_only = True,
        )
        
        if l_block_idx in (0, 1, 2): self.l_block_idx = l_block_idx
        else: raise ValueError("l_block_idx must be 0, 1, or 2.")
        
        if h_block_idx in (4, 6): self.h_block_idx = h_block_idx
        else: raise ValueError("h_block_idx must be 4 or 6.")
        
    def forward(self, x):
    
        return self.backbone(x)
