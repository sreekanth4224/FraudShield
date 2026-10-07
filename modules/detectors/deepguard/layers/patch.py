import torch
import torch.nn as nn
from typing import Type, Optional, List
from .reducesize import ReduceSize
from .se import SE
from .weight_init import trunc_normal_

class MBConvBlock(nn.Module):

    def __init__(self, 
                 dim: int,
             ):
        super().__init__()
        
        self.mb_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                      groups=dim, bias=False),
            nn.GELU(),
            SE(dim),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
        )
        
    def forward(self, x):
        x = x.contiguous()
        x =  x + self.mb_conv(x)

        return x

class PatchEmbed(nn.Module):
    
    def __init__(
                self,
                in_chs: int,
                dim: int,
                block_idx: int,
                input_resolution: List[int],
                norm_layer: Type[nn.Module] = nn.LayerNorm,
                ):
        super().__init__()
        
        H, W = input_resolution
        self.cls_token = nn.Parameter(torch.zeros(1,1,dim))
        
        if block_idx == 0:
            self.proj = nn.Sequential(
                nn.Conv2d(in_chs, dim, kernel_size=3, stride=2, padding=1),
                MBConvBlock(dim),
                nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)
            )
            n_h, n_w = H // 4, W // 4
            
        elif block_idx == 1 or block_idx == 4:
            self.proj = nn.Sequential(
                nn.Conv2d(in_chs, dim, kernel_size=3, stride=2, padding=1),
                MBConvBlock(dim)
            )
            n_h, n_w = H // 2, W // 2
            
        elif block_idx == 2 or block_idx == 6:
            self.proj = nn.Sequential(
                nn.Conv2d(in_chs, dim, kernel_size=1, stride=1, padding=0),
                MBConvBlock(dim)
            )
            n_h, n_w = H, W
            
        self.pos_embed = nn.Parameter(torch.zeros(1, n_h * n_w + 1, dim))
        self.norm = norm_layer(dim)    
        
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        x = self.proj(x)
        
        B,C,H,W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, -1, C)
        
        cls_token = self.cls_token.expand(B,-1,-1)
        x = torch.cat((cls_token, x), dim=1)
        
        x += self.pos_embed
        
        return self.norm(x.contiguous()) 

class GCViTPatchEmbed(nn.Module):
    
    def __init__(
                self,
                in_chs: int,
                dim: int,
                norm_layer: Type[nn.Module] = nn.LayerNorm,
                ):
        super().__init__()
        
        self.proj = nn.Sequential(
            nn.Conv2d(in_chs, dim, kernel_size=1, stride=1, padding=0),
            MBConvBlock(dim)
        )

        self.norm = norm_layer(dim)    
          
    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0,2,3,1)
            
        return self.norm(x)
