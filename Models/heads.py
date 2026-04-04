import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import mask_logits
from .Initializations import uniform_


class Pointer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        w1 = torch.empty(d_model * 2)
        w2 = torch.empty(d_model * 2)
        lim = 3.0 / (2.0 * d_model)
        uniform_(w1, -math.sqrt(lim), math.sqrt(lim))
        uniform_(w2, -math.sqrt(lim), math.sqrt(lim))
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)

    def forward(self, M1: torch.Tensor, M2: torch.Tensor, M3: torch.Tensor, mask: torch.Tensor):
        #X1 = torch.cat([M1, M2], dim=0)  # [B, 2C, L]
        X1 = torch.cat([M1, M2], dim=1)  # [B, 2C, L]
        X2 = torch.cat([M1, M3], dim=1)  # [B, 2C, L]
        Y1 = torch.matmul(self.w1, X1)  # [B, L]
        Y2 = torch.matmul(self.w2, X2)  # [B, L]
        Y1 = mask_logits(Y1, mask)
        Y2 = mask_logits(Y2, mask)
        p1 = F.log_softmax(Y1, dim=1)
        p2 = F.log_softmax(Y2, dim=1)
        return p1, p2

# MIGHT BE USELESS
# class Pointer(nn.Module):
#     def __init__(self, d_model: int):
#         super().__init__()
#         # 初始化权重
#         w1 = torch.empty(d_model * 2)
#         w2 = torch.empty(d_model * 2)
#         lim = 3.0 / (2.0 * d_model)
#         uniform_(w1, -math.sqrt(lim), math.sqrt(lim))
#         uniform_(w2, -math.sqrt(lim), math.sqrt(lim))
#         self.w1 = nn.Parameter(w1)
#         self.w2 = nn.Parameter(w2)

#     def forward(self, M1: torch.Tensor, M2: torch.Tensor, M3: torch.Tensor, mask: torch.Tensor):
#         # 假设输入 M1, M2, M3 形状为 [Batch, Length, Dim] (这是最常见的)
        
#         # 1. 转置为 [Batch, Dim, Length] 以便进行矩阵乘法
#         M1 = M1.transpose(1, 2)
#         M2 = M2.transpose(1, 2)
#         M3 = M3.transpose(1, 2)
        
#         # 2. 拼接 [Batch, Dim*2, Length]
#         X1 = torch.cat([M1, M2], dim=1)
#         X2 = torch.cat([M1, M3], dim=1)
        
#         # 3. 矩阵乘法 [Batch, Length]
#         # unsqueeze(1) 把 w1 变成 [1, Dim*2, 1] 以便广播
#         Y1 = torch.matmul(self.w1.unsqueeze(0).unsqueeze(2), X1.unsqueeze(1)).squeeze(1).squeeze(1)
#         # 或者更简单的写法：
#         # Y1 = torch.einsum('bdl,d->bl', X1, self.w1) 
        
#         # 让我们用最基础的写法，避免 einsum 兼容性：
#         # X1: [B, 2D, L], w1: [2D]
#         # 我们需要做的是 w1 点乘 X1 的 d 维度
#         Y1 = torch.sum(X1 * self.w1.unsqueeze(1).unsqueeze(2), dim=1) # [B, L]
#         Y2 = torch.sum(X2 * self.w2.unsqueeze(1).unsqueeze(2), dim=1) # [B, L]

#         # 4. Mask
#         Y1 = mask_logits(Y1, mask)
#         Y2 = mask_logits(Y2, mask)
        
#         # 5. Softmax (对 Length 维度，即 dim=1)
#         p1 = F.log_softmax(Y1, dim=1)
#         p2 = F.log_softmax(Y2, dim=1)
        
#         return p1, p2