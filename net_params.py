import math
import torch
import torch.nn.functional as F

from torch import nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class AdaInsNorm(nn.Module):
    def __init__(self, n_embd, n_hid, dropout=0.):
        super().__init__()
        self.emb = SinusoidalPosEmb(n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Sequential(nn.Linear(n_embd, n_hid),
                                    nn.SiLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(n_hid, n_embd*2))
        self.instancenorm = nn.InstanceNorm1d(n_embd)

    def forward(self, x, index=None, index_range=None):
        assert index or index_range
        
        if index is None:
            index_instance = torch.arange(index_range, device=x.device, dtype=torch.long).unsqueeze(0)
            index_tensor = index_instance.repeat(x.shape[0], 1).reshape(-1)
            emb = self.emb(index_tensor)
            emb = self.linear(self.silu(emb)).reshape(x.shape[0], index_range, -1)
            scale, shift = torch.chunk(emb, 2, dim=-1)
            x = self.instancenorm(x).unsqueeze(1) * (1 + scale) + shift
        else:
            index_tensor = torch.full((x.shape[0],), index, device=x.device, dtype=torch.long)
            emb = self.emb(index_tensor)
            emb = self.linear(self.silu(emb))
            scale, shift = torch.chunk(emb, 2, dim=1)
            x = self.instancenorm(x) * (1 + scale) + shift
        
        return x
    

class MLPBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        dropout=0.
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.proj(x)
    

class inverseNet(nn.Module):
    def __init__(
        self, 
        n_layers, 
        bucket_dim, 
        hash_dim,
        hidden_dim, 
        out_dim,
        share_dim=1024,
        dropout=0.1,
        layer_dim=64
    ):
        super().__init__()

        self.bn, self.hn = bucket_dim, hash_dim
        self.encoder = MLPBlock(bucket_dim, hidden_dim, hidden_dim, dropout)
        layers = [MLPBlock(hidden_dim*hash_dim, layer_dim, hidden_dim, dropout)]
        layers += [MLPBlock(hidden_dim, layer_dim, hidden_dim) for i in range(n_layers)]
        self.layers = nn.Sequential(*layers)

        self.share_dim = share_dim
        self.reset(out_dim)
        self.index_emb = AdaInsNorm(hidden_dim, layer_dim)

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, layer_dim),
            nn.Dropout(dropout),
            nn.Linear(layer_dim, share_dim),
            nn.Sigmoid()
        )

    def reset(self, out_dim):
        self.out_dim = out_dim
        self.index_range = out_dim // self.share_dim + 1
    
    def forward(self, x):
        batch = x.shape[0]
        x = self.encoder(x.reshape(batch, self.hn, self.bn)).reshape(batch, -1)
        x = self.index_emb(self.layers[0](x), index_range=self.index_range)
        for layer_idx in range(1, len(self.layers)):
            x = self.layers[layer_idx](x) + x

        x = self.out_proj(x)
        return x.reshape(batch, -1)[:, :self.out_dim]


class inverseNet_ablation(nn.Module):
    def __init__(
        self, 
        n_layers, 
        bucket_dim, 
        hash_dim,
        hidden_dim, 
        out_dim,
        share_dim=1024,
        dropout=0.1,
        layer_dim=64
    ):
        super().__init__()

        self.bn, self.hn = bucket_dim, hash_dim
        self.encoder = MLPBlock(bucket_dim, hidden_dim, hidden_dim, dropout)
        layers = [MLPBlock(hidden_dim*hash_dim, layer_dim, hidden_dim, dropout)]
        layers += [MLPBlock(hidden_dim, layer_dim, hidden_dim) for i in range(n_layers)]
        self.layers = nn.Sequential(*layers)

        self.share_dim = share_dim
        self.reset(out_dim)

        out_blocks = [nn.Linear(hidden_dim, share_dim) for _ in range(out_dim // share_dim + 1)]
        self.out_blocks = nn.Sequential(*out_blocks)
        self.act = nn.Sigmoid()

    def reset(self, out_dim):
        self.out_dim = out_dim
        self.index_range = out_dim // self.share_dim + 1
    
    def forward(self, x):
        batch = x.shape[0]
        x = self.encoder(x.reshape(batch, self.hn, self.bn)).reshape(batch, -1)
        x = self.layers[0](x)
        for layer_idx in range(1, len(self.layers)):
            x = self.layers[layer_idx](x) + x

        outputs = []

        for block_idx in range(len(self.out_blocks)):
            y = self.out_blocks[block_idx](x)
            outputs.append(y)
        
        return self.act(torch.concat(outputs, dim=-1)).reshape(batch, -1)[:, :self.out_dim]