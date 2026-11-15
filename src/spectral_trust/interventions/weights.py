# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

r"""
Weight-spectrum steering and the weight SNR diagnostic.

:func:`svd_scale_matrix` rescales the singular values of a weight matrix
``W = U \Sigma V^T`` by

.. math::
    \sigma_i' = \sigma_i \Big(1 + \alpha \, \frac{\sigma_i - \mu_\sigma}{s_\sigma}\Big)

with ``mu`` and ``s`` the mean and (unbiased) standard deviation of the
singular values, and rebuilds ``W' = U \Sigma' V^T``. ``alpha < 0``
compresses the dominant directions and ``alpha > 0`` amplifies them. This is
the spectral-steering rescaling, computed as its reference code computes it:
a float32 thin SVD on the CPU (``torch.linalg.svd(..., full_matrices=False)``)
and ``U @ diag(S') @ Vh``, cast back to the weight's dtype. The rebuilt
matrix equals the reference bit for bit; the reference then replaces the
module by a bias-free ``nn.Linear``, which is the same model only for
bias-free ``nn.Linear`` projections (the Llama, Qwen, Mistral and Gemma
``mlp.down_proj``). :meth:`WeightSite.svd_scale` edits the weight in place and
keeps the bias, and it handles GPT-2's ``Conv1D``, which the reference does
not.

:func:`weight_snr` is ``sigma_1 / median(sigma)`` over the full singular
spectrum in float32 (``torch.linalg.svdvals``; ``torch.median`` takes the
lower middle value for an even count), the convention of the spectral probe.
``weight_snr(W, k)`` keeps the fast estimate of the unreleased 0.2.3 branch
(top ``max(k, 16)`` singular values from ``torch.svd_lowrank``, the median
taken among them), which spectral-steering's layer search used, and
:func:`weight_svd_full` its full SVD; with them that code runs on released
spectral-trust.

Both read a module's weight in its ``nn.Linear`` orientation ``[out, in]``;
GPT-2's ``Conv1D`` stores ``[in, out]`` and is transposed on the way in and
out, so the same function of the input is edited on every architecture.
"""
import warnings
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import nn


def linear_weight(module: nn.Module) -> torch.Tensor:
    """The module's weight as ``[out, in]`` (a view; ``Conv1D`` transposed)."""
    w = getattr(module, "weight", None)
    if w is None or w.dim() != 2:
        raise ValueError(f"{type(module).__name__} has no 2-D weight")
    if hasattr(module, "quant_state") or hasattr(w, "quant_state"):
        raise ValueError("quantised weights are not supported; load the model in "
                         "float32, bfloat16 or float16")
    return w.T if hasattr(module, "nf") and not isinstance(module, nn.Linear) else w


def svd_scale_matrix(W: torch.Tensor, alpha: float, k: Optional[int] = None) -> torch.Tensor:
    """Singular-value rescaling of a matrix ``[out, in]`` (see the module
    notes). ``k`` rescales only the top ``k`` singular values and leaves the
    others as they are (the mean and standard deviation are still those of
    the full spectrum); ``k=None`` rescales all of them. Returns a new
    tensor in ``W``'s dtype and on its device.

    The factor ``1 + alpha * z_i`` (``z_i`` the singular value's z-score)
    must stay positive for the edit to compress (``alpha < 0``) or amplify
    (``alpha > 0``) the dominant directions. Past ``alpha = -1 / z_max`` the
    largest singular value changes sign and grows instead; a warning is
    issued. The formula is applied as written, as in the reference code."""
    U, S, Vh = torch.linalg.svd(W.detach().float().cpu(), full_matrices=False)
    factor = 1 + alpha * (S - S.mean()) / S.std()
    S_new = S * factor
    if k is not None:
        k = int(k)
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        S_new = torch.cat([S_new[:k], S[k:]])
        factor = factor[:k]
    if factor.numel() and float(factor.min()) <= 0:
        warnings.warn(
            f"svd_scale(alpha={alpha}): {int((factor <= 0).sum())} singular values get a "
            f"non-positive factor 1 + alpha * z (min {float(factor.min()):.2f}), so they flip "
            f"sign and the edit stops compressing; |alpha| below "
            f"{1.0 / float(((S - S.mean()) / S.std()).abs().max()):.3f} keeps every factor "
            f"positive", stacklevel=2)
    W_new = U @ torch.diag(S_new) @ Vh
    return W_new.to(W.dtype).to(W.device)


def matrix_snr(W: torch.Tensor) -> float:
    """``sigma_1 / median(sigma)`` of a matrix, float32, full spectrum."""
    sigma = torch.linalg.svdvals(W.detach().to(torch.float32)).cpu()
    return float(sigma[0] / sigma.median())


def lowrank_snr(W: torch.Tensor, k: int = 1) -> float:
    """The 0.2.3 branch's fast estimate: ``S[0] / S[len(S) // 2]`` of the top
    ``max(k, 16)`` singular values from ``torch.svd_lowrank`` (``niter =
    max(4, k // 4)``). The median is that of the computed top values, not of
    the full spectrum, so it is larger than :func:`matrix_snr`'s and the
    estimate is smaller; ``torch.svd_lowrank`` is randomised (seed torch for
    a fixed value)."""
    W_fp = W.detach().float()
    niter = max(4, int(k) // 4)
    _, S, _ = torch.svd_lowrank(W_fp, q=max(int(k), 16), niter=niter)
    sigma_1 = S[0].item()
    sigma_median = S[len(S) // 2].item()
    return sigma_1 / sigma_median if sigma_median > 1e-12 else 0.0


def weight_svd_full(W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full thin SVD ``(U, S, Vh)`` of a weight matrix in float32
    (``torch.linalg.svd(W.float(), full_matrices=False)``), as in the 0.2.3
    branch. Added in 0.5.0."""
    return torch.linalg.svd(W.float(), full_matrices=False)


def weight_snr(model, layer: Optional[int] = None, module: str = "mlp.down_proj",
               k: Optional[int] = None) -> Union[float, np.ndarray]:
    """Spectral SNR ``sigma_1 / median(sigma)`` of a weight matrix.

    Two forms:

    - ``weight_snr(model, layer, module)``: the matrix at ``module`` (dotted
      path inside block ``layer``; ``"mlp.down_proj"`` resolves to the MLP
      output projection of architectures that name it differently, such as
      GPT-2's ``mlp.c_proj``), over the full spectrum. ``layer=None`` returns
      one value per layer as a numpy array, a layer sweep.
    - ``weight_snr(W)`` or ``weight_snr(W, k)``: a 2-D weight tensor. Without
      ``k``, the full spectrum; with ``k``, the fast top-k estimate of the
      0.2.3 branch (:func:`lowrank_snr`), which is what spectral-steering's
      layer search called as ``weight_snr(W, k=32)``.

    Args:
        model: a :class:`~spectral_trust.SpectralModel`, a Hugging Face
            model, or a 2-D weight tensor.
        layer: block index for a model. For a tensor, a second positional
            argument is taken as ``k``.
        module: dotted path inside the block (model form).
        k: number of top singular values for the fast estimate (tensor
            form); None uses the full spectrum.

    Added in 0.5.0.
    """
    if torch.is_tensor(model):
        if model.dim() != 2:
            raise ValueError(f"expected a 2-D weight, got shape {tuple(model.shape)}")
        if k is None and layer is not None:
            k = layer
        return matrix_snr(model) if k is None else lowrank_snr(model, int(k))
    from ._arch import Arch
    arch = Arch(model)
    snr = matrix_snr if k is None else (lambda W: lowrank_snr(W, int(k)))
    if layer is None:
        return np.array([snr(linear_weight(arch.module(i, module)[1]))
                         for i in range(arch.n_layers)])
    return snr(linear_weight(arch.module(layer, module)[1]))
