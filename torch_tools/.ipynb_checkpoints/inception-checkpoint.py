import torch
from torch import nn, Tensor
import torch.nn.functional as F 
from typing import Callable, Any, Optional, Tuple, List

class BasicConv3d(nn.Module):
    """Implements a simple 3d conv module"""
    
    def __init__(self, in_channels: int,  out_channels: int, **kwargs: Any) -> None:
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, eps=0.001)
        
    def forward(self, x: Tensor) -> Tensor: 
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    
class InceptionA(nn.Module):
    # require conv_block to be a callable type that returns a nn.module object 
    def __init__(self, 
                 in_channels: int, 
                 pool_features: int,
                 conv_block: Optional[Callable[..., nn.Module]] = None,
                 **kwargs: Any) -> None:
        super(InceptionA, self).__init__()
        
        # if no specific conv_block is passed then use default
        if conv_block is None: 
            conv_block = BasicConv3d
        
        # small conv _branch
        self.branch1 = conv_block(in_channels, 64, kernel_size=1)
        
        # medium conv branch 
        self.branch3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)
        
        # large conv branch
        self.branch5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5_2 = conv_block(48, 64, kernel_size=5, padding=2) 
    
        # downsample pooling branch
        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)       
        
    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1 = self.branch1(x)
        
        branch3 = self.branch3dbl_1(x)
        branch3 = self.branch3dbl_2(branch3)
        branch3 = self.branch3dbl_3(branch3)
        
        branch5 = self.branch5_1(x)
        branch5 = self.branch5_2(branch5)
        
        branch_pool = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        
        outputs = [branch1, branch3, branch5, branch_pool]
        return outputs
    
    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)
        
        