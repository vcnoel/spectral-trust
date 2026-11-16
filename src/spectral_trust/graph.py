# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

import torch
from typing import Optional
import logging
from .config import GSPConfig

logger = logging.getLogger(__name__)

#: Accepted ``GSPConfig.edge_weights``: the attention weights, the norm of
#: what each token adds to the attention output (summed over heads inside
#: the norm), or that norm per head (see :mod:`spectral_trust.value_weighted`).
EDGE_WEIGHTS = ("attention", "value", "value_per_head")

#: Accepted ``GSPConfig.graph_direction``.
GRAPH_DIRECTIONS = ("symmetric", "directed")


def check_graph_options(config) -> None:
    """Raise on an unknown ``edge_weights`` or ``graph_direction``."""
    edge = getattr(config, "edge_weights", "attention")
    if edge not in EDGE_WEIGHTS:
        raise ValueError(f"edge_weights must be one of {EDGE_WEIGHTS}, got {edge!r}")
    direction = getattr(config, "graph_direction", "symmetric")
    if direction not in GRAPH_DIRECTIONS:
        raise ValueError(f"graph_direction must be one of {GRAPH_DIRECTIONS}, got {direction!r}")


def default_graph(config) -> bool:
    """True when the config builds the graph of every earlier release:
    attention edges, symmetrised. Checks the options first."""
    check_graph_options(config)
    return (getattr(config, "edge_weights", "attention") == "attention"
            and getattr(config, "graph_direction", "symmetric") == "symmetric")


class GraphConstructor:
    """Constructs dynamic attention graphs from transformer attention patterns"""
    
    def __init__(self, config: GSPConfig):
        self.config = config

    def layer_graph(self, edges: torch.Tensor,
                    attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The head-averaged graph of a layer, as ``config`` defines it.

        With ``graph_direction="symmetric"`` (the default) it is
        ``aggregate_heads(symmetrize_attention(edges))``, the adjacency every
        measurement of the library builds. With ``"directed"`` it is
        ``directed_operator(aggregate_heads(edges))``: heads are aggregated
        without symmetrisation and each row is divided by its sum, which
        gives the causal row-normalised operator ``P``
        (:func:`spectral_trust.directed.directed_operator`;
        ``symmetrization`` does not apply).

        Args:
            edges: ``[B, H, S, S]`` per-head edge weights (attention, or
                :func:`~spectral_trust.value_weighted.value_weighted_edges`
                with ``per_head=True``), or ``[B, S, S]`` for one edge matrix
                per item (value-weighted edges summed over heads), row i
                reading column j.
            attn_mask: as in :meth:`aggregate_heads`.

        Returns:
            ``[B, S, S]``.
        """
        check_graph_options(self.config)
        if edges.dim() == 3:
            edges = edges.unsqueeze(1)
        if getattr(self.config, "graph_direction", "symmetric") == "directed":
            return self.directed_operator(self.aggregate_heads(edges, attn_mask=attn_mask))
        return self.aggregate_heads(self.symmetrize_attention(edges), attn_mask=attn_mask)

    def directed_operator(self, adjacency: torch.Tensor) -> torch.Tensor:
        """Row-normalised operator ``P = D_out^-1 A`` of a directed adjacency
        (``config.remove_self_loops`` applies); see
        :func:`spectral_trust.directed.directed_operator`."""
        from .directed import directed_operator
        return directed_operator(adjacency, remove_self_loops=self.config.remove_self_loops)

    def magnetic_laplacian(self, adjacency: torch.Tensor, q: Optional[float] = None,
                           dtype: torch.dtype = torch.complex128) -> torch.Tensor:
        """Magnetic Laplacian of a directed adjacency, normalised for
        ``normalization="sym"`` and unnormalised for ``"none"``, with charge
        ``q`` (default ``config.magnetic_q``); see
        :func:`spectral_trust.directed.magnetic_laplacian`."""
        from .directed import magnetic_laplacian, magnetic_normalized
        q = getattr(self.config, "magnetic_q", 0.25) if q is None else q
        return magnetic_laplacian(adjacency, q=q,
                                  normalized=magnetic_normalized(self.config.normalization),
                                  remove_self_loops=self.config.remove_self_loops, dtype=dtype)
        
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
                # Ensure mask is on the same device as attention
                attn_mask = attn_mask.to(A.device)
                
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
        Construct graph Laplacian from adjacency matrix. Negative weights are
        set to zero: signed graphs are not supported yet.
        Args:
            adjacency: [batch, seq_len, seq_len] adjacency matrix
        Returns:
            Graph Laplacian matrix
        """
        # Ensure non-negative weights and promote to Float32 to avoid BFloat16-CPU issues
        adjacency = torch.clamp(adjacency, min=0).to(torch.float32)
        
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
            deg_inv = torch.zeros_like(degrees)
            mask = degrees > 1e-8
            deg_inv[mask] = 1.0 / degrees[mask].clamp(min=1e-8)
            # D^{-1} W by broadcasting: O(S^2), where a dense diagonal
            # matrix product is O(S^3)
            laplacian = torch.eye(adjacency.shape[-1], device=adjacency.device).unsqueeze(0) - deg_inv.unsqueeze(-1) * adjacency
        elif self.config.normalization == "sym":
            # Symmetric normalized Laplacian: L = I - D^{-1/2}WD^{-1/2}
            deg_sqrt_inv = torch.zeros_like(degrees)
            mask = degrees > 1e-8
            deg_sqrt_inv[mask] = 1.0 / torch.sqrt(degrees[mask].clamp(min=1e-8))
            # D^{-1/2} W D^{-1/2} by broadcasting: O(S^2), where two dense
            # diagonal matrix products are O(S^3)
            normalized_adj = deg_sqrt_inv.unsqueeze(-1) * adjacency * deg_sqrt_inv.unsqueeze(-2)
            laplacian = torch.eye(adjacency.shape[-1], device=adjacency.device).unsqueeze(0) - normalized_adj
        elif self.config.normalization == "none":
            # Combinatorial Laplacian: L = D - W
            degree_diag = torch.diag_embed(degrees)
            laplacian = degree_diag - adjacency
        else:
            raise ValueError(f"Unknown normalization method: {self.config.normalization}")
        
        return laplacian

    def subgraph_laplacian(self, full_adjacency: torch.Tensor, target_indices: list) -> torch.Tensor:
        """
        Extract induced subgraph adjacency and build its Laplacian.
        full_adjacency: [B, N, N]
        target_indices: list of token indices to isolate
        """
        if not target_indices:
            return self.construct_laplacian(full_adjacency)
        indices = torch.tensor(target_indices, device=full_adjacency.device)
        sub_adj = full_adjacency.index_select(-2, indices).index_select(-1, indices)
        return self.construct_laplacian(sub_adj)


def _index(indices, device) -> torch.Tensor:
    return torch.as_tensor(list(indices), dtype=torch.long, device=device)


def drop_and_renormalize(attention: torch.Tensor, drop=0, *, keep=None) -> torch.Tensor:
    """Attention with some tokens removed and every remaining row divided by
    its new sum, so each query again spreads unit mass over the kept keys.

    This is the BOS-drop construction of Bao et al. (2026): drop the sink
    (``drop=0``) and renormalise the rows. ``GSPConfig.subgraph_indices``
    and ``token_span`` drop nodes without renormalising; with
    ``GSPConfig.subgraph_renormalize=True`` they renormalise as here.

    Args:
        attention: ``[..., S, S]`` attention weights, row i the query.
        drop: index or indices of the tokens to remove (default the first
            token). Ignored when ``keep`` is given.
        keep: indices of the tokens to keep, in order.

    Returns:
        ``[..., k, k]`` with the kept tokens. A row whose kept entries sum to
        zero stays zero.
    """
    S = attention.shape[-1]
    if keep is None:
        gone = {int(d) % S for d in ([drop] if isinstance(drop, int) else drop)}
        keep = [i for i in range(S) if i not in gone]
    idx = _index(keep, attention.device)
    sub = attention.index_select(-2, idx).index_select(-1, idx)
    total = sub.sum(dim=-1, keepdim=True)
    return torch.where(total > 0, sub / torch.where(total > 0, total, torch.ones_like(total)),
                       sub)


def renormalized_within(attention: torch.Tensor, keep) -> torch.Tensor:
    """``attention`` of the same shape, zero outside the kept rows and
    columns, with the kept block equal to :func:`drop_and_renormalize`
    ``(attention, keep=keep)``. The induced subgraph of any graph built from
    it is the graph of the renormalised block."""
    idx = _index(keep, attention.device)
    out = torch.zeros_like(attention)
    block = drop_and_renormalize(attention, keep=keep)
    rows = out.index_select(-2, idx)
    rows.index_copy_(-1, idx, block)
    return out.index_copy(-2, idx, rows)
