import math
import torch
import torch.nn.functional as F

from torch import nn


class SinusoidalPosEmb(nn.Module):
    """
    Implements sinusoidal positional embedding to encode position information.
    This is commonly used in transformer architectures to give the model awareness of token positions.
    
    Args:
        dim: Dimension of the positional embedding vector
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        Forward pass to compute sinusoidal positional embeddings
        
        Args:
            x: Input tensor containing position indices
            
        Returns:
            Positional embeddings computed using sine and cosine functions
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class AdaInsNorm(nn.Module):
    """
    Adaptive Instance Normalization layer that adapts normalization based on positional information.
    Combines instance normalization with learnable scale and shift parameters that depend on position.
    
    Args:
        n_embd: Embedding dimension size
        n_hid: Hidden layer size for transformation
        dropout: Dropout rate applied during transformation
    """
    def __init__(self, n_embd, n_hid, dropout=0.):
        super().__init__()
        self.emb = SinusoidalPosEmb(n_embd)  # Positional embedding module
        self.silu = nn.SiLU()  # SiLU activation function (Swish variant)
        # Linear transformation to generate scale and shift parameters
        self.linear = nn.Sequential(nn.Linear(n_embd, n_hid),
                                    nn.SiLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(n_hid, n_embd*2))
        self.instancenorm = nn.InstanceNorm1d(n_embd)  # Instance normalization layer

    def forward(self, x, index=None, index_range=None):
        """
        Forward pass applying adaptive instance normalization
        
        Args:
            x: Input tensor to normalize
            index: Specific index for single position processing
            index_range: Range of indices for sequence processing
            
        Returns:
            Adaptively normalized tensor with position-dependent scaling and shifting
        """
        assert index or index_range
        
        if index is None:
            # Process a range of indices for sequence normalization
            index_instance = torch.arange(index_range, device=x.device, dtype=torch.long).unsqueeze(0)
            index_tensor = index_instance.repeat(x.shape[0], 1).reshape(-1)
            emb = self.emb(index_tensor)
            emb = self.linear(self.silu(emb)).reshape(x.shape[0], index_range, -1)
            scale, shift = torch.chunk(emb, 2, dim=-1)
            x = self.instancenorm(x).unsqueeze(1) * (1 + scale) + shift
        else:
            # Process a single index
            index_tensor = torch.full((x.shape[0],), index, device=x.device, dtype=torch.long)
            emb = self.emb(index_tensor)
            emb = self.linear(self.silu(emb))
            scale, shift = torch.chunk(emb, 2, dim=1)
            x = self.instancenorm(x) * (1 + scale) + shift
        
        return x
    

class MLPBlock(nn.Module):
    """
    Multi-Layer Perceptron block with ReLU activation and dropout.
    Used as a basic building block in the network architecture.
    
    Args:
        in_dim: Input dimension
        hidden_dim: Hidden layer dimension
        out_dim: Output dimension
        dropout: Dropout rate applied after first linear layer
    """
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        dropout=0.
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),  # First linear transformation
            nn.ReLU(),                      # ReLU activation
            nn.Dropout(dropout),           # Dropout regularization
            nn.Linear(hidden_dim, out_dim), # Second linear transformation
            nn.ReLU()                       # ReLU activation
        )
    
    def forward(self, x):
        """
        Forward pass through the MLP block
        
        Args:
            x: Input tensor
            
        Returns:
            Transformed tensor after passing through the MLP layers
        """
        return self.proj(x)
    

class inverseNet(nn.Module):
    """
    Inverse network that maps bucket values back to estimated original frequencies.
    This is the core neural network component of the UCL-sketch algorithm that learns
    to correct estimation errors in traditional sketch algorithms.
    
    Args:
        n_layers: Number of hidden layers in the network
        bucket_dim: Dimension of each bucket in the sketch
        hash_dim: Number of hash functions used in the sketch
        hidden_dim: Hidden dimension size for transformations
        out_dim: Output dimension (total number of items to estimate)
        share_dim: Shared dimension size for parameter sharing
        dropout: Dropout rate for regularization
        layer_dim: Dimension of intermediate layers
    """
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

        self.bn, self.hn = bucket_dim, hash_dim  # Store bucket and hash dimensions
        # Encoder to transform bucket values
        self.encoder = MLPBlock(bucket_dim, hidden_dim, hidden_dim, dropout)
        
        # Create network layers - first layer processes concatenated buckets, subsequent layers process hidden states
        layers = [MLPBlock(hidden_dim*hash_dim, layer_dim, hidden_dim, dropout)]
        layers += [MLPBlock(hidden_dim, layer_dim, hidden_dim) for i in range(n_layers)]
        self.layers = nn.Sequential(*layers)

        self.share_dim = share_dim  # Shared dimension for parameter efficiency
        self.reset(out_dim)  # Initialize output dimensions
        # Adaptive normalization with positional embedding
        self.index_emb = AdaInsNorm(hidden_dim, layer_dim)

        # Final projection to output space with sigmoid activation
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, layer_dim),
            nn.Dropout(dropout),
            nn.Linear(layer_dim, share_dim),
            nn.Sigmoid()
        )

    def reset(self, out_dim):
        """
        Reset the output dimension parameters
        
        Args:
            out_dim: New output dimension to set
        """
        self.out_dim = out_dim
        self.index_range = out_dim // self.share_dim + 1
    
    def forward(self, x):
        """
        Forward pass to estimate original frequencies from sketch buckets
        
        Args:
            x: Input tensor of bucket values from the sketch
            
        Returns:
            Estimated original frequency values for each item
        """
        batch = x.shape[0]
        # Reshape input to process each hash table separately, then encode
        x = self.encoder(x.reshape(batch, self.hn, self.bn)).reshape(batch, -1)
        # Apply first layer and add positional embedding
        x = self.index_emb(self.layers[0](x), index_range=self.index_range)
        # Apply remaining layers with residual connections
        for layer_idx in range(1, len(self.layers)):
            x = self.layers[layer_idx](x) + x

        # Project to final output space
        x = self.out_proj(x)
        return x.reshape(batch, -1)[:, :self.out_dim]


class inverseNet_ablation(nn.Module):
    """
    Ablation version of inverseNet that uses separate linear blocks for output instead of shared parameters.
    This version allows comparison to see the effect of parameter sharing strategy.
    
    Args:
        n_layers: Number of hidden layers in the network
        bucket_dim: Dimension of each bucket in the sketch
        hash_dim: Number of hash functions used in the sketch
        hidden_dim: Hidden dimension size for transformations
        out_dim: Output dimension (total number of items to estimate)
        share_dim: Shared dimension size for parameter sharing
        dropout: Dropout rate for regularization
        layer_dim: Dimension of intermediate layers
    """
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

        self.bn, self.hn = bucket_dim, hash_dim  # Store bucket and hash dimensions
        # Encoder to transform bucket values
        self.encoder = MLPBlock(bucket_dim, hidden_dim, hidden_dim, dropout)
        
        # Create network layers - first layer processes concatenated buckets, subsequent layers process hidden states
        layers = [MLPBlock(hidden_dim*hash_dim, layer_dim, hidden_dim, dropout)]
        layers += [MLPBlock(hidden_dim, layer_dim, hidden_dim) for i in range(n_layers)]
        self.layers = nn.Sequential(*layers)

        self.share_dim = share_dim
        self.reset(out_dim)

        # Create separate output blocks for each chunk of outputs
        out_blocks = [nn.Linear(hidden_dim, share_dim) for _ in range(out_dim // share_dim + 1)]
        self.out_blocks = nn.Sequential(*out_blocks)
        self.act = nn.Sigmoid()  # Sigmoid activation for output

    def reset(self, out_dim):
        """
        Reset the output dimension parameters
        
        Args:
            out_dim: New output dimension to set
        """
        self.out_dim = out_dim
        self.index_range = out_dim // self.share_dim + 1
    
    def forward(self, x):
        """
        Forward pass to estimate original frequencies from sketch buckets
        
        Args:
            x: Input tensor of bucket values from the sketch
            
        Returns:
            Estimated original frequency values for each item
        """
        batch = x.shape[0]
        # Reshape input to process each hash table separately, then encode
        x = self.encoder(x.reshape(batch, self.hn, self.bn)).reshape(batch, -1)
        # Apply first layer
        x = self.layers[0](x)
        # Apply remaining layers with residual connections
        for layer_idx in range(1, len(self.layers)):
            x = self.layers[layer_idx](x) + x

        outputs = []

        # Process each output block separately
        for block_idx in range(len(self.out_blocks)):
            y = self.out_blocks[block_idx](x)
            outputs.append(y)
        
        # Concatenate all outputs and apply activation
        return self.act(torch.concat(outputs, dim=-1)).reshape(batch, -1)[:, :self.out_dim]