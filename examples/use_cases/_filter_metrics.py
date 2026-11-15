# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""Metrics of filter_side_effects.py: next-token changes, graph measures and
greedy-generation quality."""
import torch


def wiki_rows(tok, n, length):
    """The first ``n`` WikiText-2 test lines that are not headings and have at
    least ``length`` tokens, cut to ``length`` tokens."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    rows = []
    for line in ds["text"]:
        s = line.strip()
        if s and not s.startswith("="):
            ids = tok(s)["input_ids"]
            if len(ids) >= length:
                rows.append(ids[:length])
        if len(rows) == n:
            break
    return torch.tensor(rows)


def lam2_sink(attentions):
    """Per layer: Fiedler value of the head-averaged symmetrised graph (sym
    Laplacian, float64, all tokens) and sink mass (attention to position 0,
    mean over queries 1..). Returns two ``[B, L]`` arrays."""
    lam, sink = [], []
    for a in attentions:
        Ab = a.double().mean(1).cpu()
        W = 0.5 * (Ab + Ab.mT)
        r = W.sum(-1).clamp_min(1e-12).rsqrt()
        L = torch.eye(W.shape[-1], dtype=W.dtype) - r[..., :, None] * W * r[..., None, :]
        lam.append(torch.linalg.eigvalsh(L)[:, 1])
        sink.append(Ab[:, 1:, 0].mean(-1))
    return torch.stack(lam, 1).numpy(), torch.stack(sink, 1).numpy()


def token_metrics(lp_clean, logits, ids):
    """KL(clean || edited) ``[B, S]``, top-1 agreement ``[B, S]`` and mean
    next-token NLL per item, clean and edited ``[B]``."""
    lpc = lp_clean.to(logits.device)
    lpe = logits.float().log_softmax(-1)
    kl = (lpc.exp() * (lpc - lpe)).sum(-1)
    top = (lpc.argmax(-1) == lpe.argmax(-1)).float()
    tgt = ids[:, 1:].unsqueeze(-1).to(lpc.device)
    nc = -lpc[:, :-1].gather(-1, tgt).squeeze(-1).mean(-1)
    ne = -lpe[:, :-1].gather(-1, tgt).squeeze(-1).mean(-1)
    return [t.cpu().numpy() for t in (kl, top, nc, ne)]


@torch.no_grad()
def greedy(model, prompts, n_new):
    """Greedy continuation without a KV cache (each step recomputes the
    sequence, so a causal edit sees the same prefix as in a full pass)."""
    ids = prompts.clone()
    for _ in range(n_new):
        out = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[:, -1]
        ids = torch.cat([ids, out.argmax(-1, keepdim=True)], 1)
    return ids[:, prompts.shape[1]:].cpu()


def gen_stats(tok, gen, ref):
    """Per continuation: repetition rate (1 - unique/total tokens), distinct-2,
    largest single-token share, alphanumeric share of non-space characters,
    the first position that differs from the unedited continuation, and the
    degenerate flag: distinct-2 < 0.5 (a loop), or one token > 40% of the
    tokens, or alphanumeric share < 0.5 (punctuation or symbol runs)."""
    out = []
    for g, r in zip(gen.tolist(), ref.tolist()):
        text = tok.decode(g)
        chars = [c for c in text if not c.isspace()]
        alnum = sum(c.isalnum() for c in chars) / max(len(chars), 1)
        d2 = len(set(zip(g, g[1:]))) / max(len(g) - 1, 1)
        share = max(g.count(t) for t in set(g)) / len(g)
        first = next((i for i, (a, b) in enumerate(zip(g, r)) if a != b), len(g))
        out.append(dict(text=text, rep=1 - len(set(g)) / len(g), distinct2=d2,
                        max_share=share, alnum=alnum, first_diff=first,
                        degenerate=bool(d2 < 0.5 or share > 0.4 or alnum < 0.5)))
    return out
