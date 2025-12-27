import torch
from typing import Optional
import logging
from .config import GSPConfig

logger = logging.getLogger(__name__)

class GraphConstructor:
    """Constructs dynamic attention graphs from transformer attention patterns"""
    
    def __init__(self, config: GSPConfig):
        self.config = config
        
    def symmetrize_attention(self, attention: torch.Tensor) -> torch.Tensor:
        """
        Symmetrize attention matrix according to specified method
        Args:
            attention: [batch, heads, seq_len, seq_len] attention tensor
        Returns:
            Symmetrized attention tensor
        """
        if self.config.symmetrization == "symmetric":
            return 0.5 * (attention + attention.transpose(-2, -1))
        elif self.config.symmetrization == "row_norm":
            row_sums = attention.sum(dim=-1, keepdim=True)
            attention_norm = attention / (row_sums + 1e-8)
            return 0.5 * (attention_norm + attention_norm.transpose(-2, -1))
        elif self.config.symmetrization == "col_norm":
            col_sums = attention.sum(dim=-2, keepdim=True)
            attention_norm = attention / (col_sums + 1e-8)
            return 0.5 * (attention_norm + attention_norm.transpose(-2, -1))
        else:
            raise ValueError(f"Unknown symmetrization method: {self.config.symmetrization}")
    
    def aggregate_heads(
        self,
        attention: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,  # [B,Q] or broadcastable to [B,1,Q,K]
    ) -> torch.Tensor:
        """
        Aggregate multi-head attention into single adjacency matrix.
        attention:  [B, H, Q, K]
        attn_mask:  if provided, True for real tokens. Either [B,Q] or broadcastable
                    to [B,1,Q,K]. Used only for weighting in 'attention_weighted'.
        returns:    [B, Q, K]
        """
        if attention.dim() != 4:
            raise ValueError(f"Expected attention [B,H,Q,K], got {list(attention.shape)}")

        B, H, Q, K = attention.shape

        method = self.config.head_aggregation

        if method == "uniform":
            # Per-batch uniform weights over heads
            weights = attention.new_full((B, H), 1.0 / H)

        elif method == "attention_weighted":
            A = attention

            # Optional masking so padded tokens don't influence head weights
            if attn_mask is not None:
                if attn_mask.dim() == 2:           # [B,Q] -> build keep mask [B,1,Q,K]
                    qmask = attn_mask.unsqueeze(1).unsqueeze(-1)  # [B,1,Q,1]
                    kmask = attn_mask.unsqueeze(1).unsqueeze(2)   # [B,1,1,K]
                    keep = qmask & kmask                          # [B,1,Q,K]
                else:
                    keep = attn_mask                               # already broadcastable
                A = A.masked_fill(~keep, 0.0)

            # Total “mass” per head (per batch)
            masses = A.sum(dim=(2, 3))                              # [B,H]
            weights = masses / (masses.sum(dim=1, keepdim=True) + 1e-8)

        elif method == "learnable":
            # Not implemented; fall back to uniform but keep correct shape
            logger.warning("Learnable aggregation not implemented; using uniform.")
            weights = attention.new_full((B, H), 1.0 / H)

        else:
            raise ValueError(f"Unknown head aggregation method: {method}")

        # IMPORTANT: broadcast weights on the HEAD axis only
        w = weights.view(B, H, 1, 1)                                # [B,H,1,1]
        aggregated = (attention * w).sum(dim=1)                     # [B,Q,K]
        return aggregated

    
    def construct_laplacian(self, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Construct graph Laplacian from adjacency matrix
        Args:
            adjacency: [batch, seq_len, seq_len] adjacency matrix
        Returns:
            Graph Laplacian matrix
        """
        # Ensure non-negative weights
        adjacency = torch.clamp(adjacency, min=0)
        
        # Remove self-loops if configured
        if self.config.remove_self_loops:
            # Assume adjacency is [Q, Q] because construct_laplacian typically handles symmetrized square matrices
            # Set diagonal to zero
            mask = torch.eye(adjacency.shape[-1], device=adjacency.device).bool()
            adjacency = adjacency.masked_fill(mask, 0.0)
        
        # Compute degree matrix
        degrees = adjacency.sum(dim=-1)  # [batch, seq_len]
        
        if self.config.normalization == "rw":
            # Random walk Laplacian: L = I - D^{-1}W
            deg_inv = torch.where(degrees > 1e-8, 1.0 / degrees, torch.zeros_like(degrees))
            deg_inv_diag = torch.diag_embed(deg_inv)
            laplacian = torch.eye(adjacency.shape[-1], device=adjacency.device).unsqueeze(0) - torch.matmul(deg_inv_diag, adjacency)
        elif self.config.normalization == "sym":
            # Symmetric normalized Laplacian: L = I - D^{-1/2}WD^{-1/2}
            deg_sqrt_inv = torch.where(degrees > 1e-8, 1.0 / torch.sqrt(degrees), torch.zeros_like(degrees))
            deg_sqrt_inv_diag = torch.diag_embed(deg_sqrt_inv)
            normalized_adj = torch.matmul(torch.matmul(deg_sqrt_inv_diag, adjacency), deg_sqrt_inv_diag)
            laplacian = torch.eye(adjacency.shape[-1], device=adjacency.device).unsqueeze(0) - normalized_adj
        elif self.config.normalization == "none":
            # Combinatorial Laplacian: L = D - W
            degree_diag = torch.diag_embed(degrees)
            laplacian = degree_diag - adjacency
        else:
            raise ValueError(f"Unknown normalization method: {self.config.normalization}")
        
        return laplacian
