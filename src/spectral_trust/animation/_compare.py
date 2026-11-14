# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""``compare``: two prompts side by side, layer by layer (for example a
clean prompt and the same prompt with an injected instruction)."""
import difflib
from typing import Optional, Sequence, Tuple

import numpy as np

from ._core import (
    Canvas, display_token, hex_rgb, layer_positions, plotly_layout, plotly_player, r4, rgba,
    split_position, write_plotly_html,
)
from ._data import graph_series, layer_graphs, match_signs, model_label, pick_layers, single
from ._graphdraw import (
    circle_positions, draw_circle_labels, draw_edges, draw_nodes, fiedler_colours, label_room,
)
from ._scene import Scene


def _pair(cache, other, texts):
    if other is not None:
        return single(cache, texts[0] if cache.is_batch else 0), single(other, 0)
    if not cache.is_batch or cache.n_texts < 2:
        raise ValueError("compare needs two texts: pass other=<second cache>, or a cache "
                         "of a batch (run_with_spectra([clean, injected]))")
    return cache.text(texts[0]), cache.text(texts[1])


class CompareData:
    """Both graphs of every layer on one set of token positions: the tokens
    of the two prompts are matched (difflib), so a shared token sits at the
    same place in both panels and the extra tokens of either open a gap in
    the other."""

    def __init__(self, cache, other=None, texts: Tuple[int, int] = (0, 1), layers=None,
                 labels: Sequence[str] = ("clean", "injected")):
        A, B = _pair(cache, other, texts)
        self.A, self.B = A, B
        common = [lay for lay in A.layers if lay in B.layers]
        if not common:
            raise ValueError("the two caches share no layer")
        if layers is None:
            self.layers = common
        else:
            self.layers = [lay for lay in pick_layers(A, layers) if lay in common]
        self.labels = tuple(labels)
        gA, gB = layer_graphs(A, self.layers), layer_graphs(B, self.layers)
        tA, tB = gA[0].tokens, gB[0].tokens
        self.tokA = [display_token(t) for t in tA]
        self.tokB = [display_token(t) for t in tB]
        entries = []
        sm = difflib.SequenceMatcher(a=list(tA), b=list(tB), autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                entries += [(i1 + k, j1 + k) for k in range(i2 - i1)]
            else:
                entries += [(i, None) for i in range(i1, i2)]
                entries += [(None, j) for j in range(j1, j2)]
        self.entries = entries
        self.n = len(entries)
        self.idxA = np.array([u for u, (a, _) in enumerate(entries) if a is not None])
        self.idxB = np.array([u for u, (_, b) in enumerate(entries) if b is not None])
        self.onlyA = np.array([b is None for a, b in entries if a is not None])
        self.onlyB = np.array([a is None for a, b in entries if b is not None])
        self.union_tokens = [self.tokA[a] if a is not None else self.tokB[b] for a, b in entries]
        self.union_onlyB = np.array([a is None for a, _ in entries])
        self.union_onlyA = np.array([b is None for _, b in entries])
        self.WA = np.stack([g.normalized() for g in gA])
        self.WB = np.stack([g.normalized() for g in gB])
        self.rawB = np.stack([g.W for g in gB])
        # Fiedler signs: A matched along the layers, B matched to A on shared tokens
        self.uA = np.stack(match_signs([g.fiedler for g in gA]))
        shared = [(a, b) for a, b in entries if a is not None and b is not None]
        ia = np.array([a for a, _ in shared], dtype=int)
        ib = np.array([b for _, b in shared], dtype=int)
        uB = []
        prev = None
        for li, g in enumerate(gB):
            u = g.fiedler.copy()
            ref = float(u[ib] @ self.uA[li][ia]) if len(ia) else 0.0
            if ref < 0 or (ref == 0 and prev is not None and float(u @ prev) < 0):
                u = -u
            uB.append(u)
            prev = u
        self.uB = np.stack(uB)
        self.alertA = self._alert(self.onlyA)
        self.alertB = self._alert(self.onlyB)
        self.lamA = graph_series(A, "fiedler_value", self.layers)
        self.lamB = graph_series(B, "fiedler_value", self.layers)
        self.hferA = graph_series(A, "hfer", self.layers)
        self.hferB = graph_series(B, "hfer", self.layers)
        # share of B's edge weight that touches the tokens only B has
        share = []
        for W in self.rawB:
            off = W.copy()
            np.fill_diagonal(off, 0.0)
            tot = off.sum()
            m = self.onlyB
            share.append(float(off[m].sum() + off[:, m].sum() - off[np.ix_(m, m)].sum())
                         / tot if tot > 0 and m.any() else np.nan)
        self.span_share = np.asarray(share)

    @staticmethod
    def _alert(only: np.ndarray) -> np.ndarray:
        return only[:, None] | only[None, :]

    def at(self, p: float):
        i, j, e = split_position(p, len(self.layers))
        lerp = lambda x: (1 - e) * x[i] + e * x[j]  # noqa: E731
        return i, j, e, lerp


class Compare(Scene):
    kind = "compare"

    def __init__(self, cache, canvas: Canvas, *, fps: float, duration: Optional[float] = None,
                 other=None, texts: Tuple[int, int] = (0, 1), layers=None,
                 labels: Sequence[str] = ("clean", "injected"), title: Optional[str] = None,
                 subtitle: Optional[str] = None, hold: float = 0.3, per_layer: float = 0.45,
                 wordmark: bool = False, seed: int = 0):
        self.data = d = CompareData(cache, other, texts, layers, labels)
        L = len(d.layers)
        if duration is None:
            duration = 1.0 + per_layer * max(L - 1, 1) + 1.5
        n = int(round(duration * fps))
        super().__init__(canvas, fps, n, wordmark)
        self.pos_t = layer_positions(n, L, int(round(1.0 * fps)), int(round(1.5 * fps)), hold)
        self.title = title if title is not None else f"{d.labels[0].capitalize()} vs {d.labels[1]}"
        self.subtitle = subtitle if subtitle is not None else (
            f"{model_label(d.A)} · shared tokens in the same place, {d.labels[1]}-only tokens "
            "and their edges in vermillion")
        cv = self.cv
        # compact frames (a GIF for phones) give the trajectories' room to the graphs
        self.traj_h = 0.0 if cv.compact else cv.px(132, 70)
        gap = cv.px(36, 14)
        pw = (cv.W - 2 * self.margin - gap) / 2
        y0, y1 = self.top + cv.px(8, 4), self.bottom - self.traj_h
        self.cap_fs = cv.fs(17, 12)
        self.label_fs = cv.fs(13, 10.5) if d.n > 16 else cv.fs(15, 11)
        radial = d.n > 16
        room = label_room(cv, d.union_tokens, self.label_fs, radial)
        cap = self.cap_fs * 1.9
        self.R = max(min(pw / 2 - room, (y1 - y0 - cap) / 2 - room * (1.0 if radial else 0.6)),
                     cv.px(40, 24))
        self.r_node = float(np.clip(cv.px(7.5, 4.5) * 20 / max(d.n, 10) * 1.4, cv.px(4, 3),
                                    cv.px(9, 6)))
        self.panels = []
        for k in range(2):
            x0 = self.margin + k * (pw + gap)
            c = (x0 + pw / 2, y0 + cap + (y1 - y0 - cap) / 2)
            self.panels.append({"x0": x0, "w": pw, "y0": y0, "centre": c,
                                "pos": circle_positions(d.n, c[0], c[1], self.R)})
        self.radial = radial

    def draw(self, f: int):
        cv, th, d = self.cv, self.th, self.data
        p = self.pos_t[f]
        i, j, e, lerp = d.at(p)
        lay = d.layers[i] if e < 0.5 else d.layers[j]
        for k, panel in enumerate(self.panels):
            W = lerp(d.WA if k == 0 else d.WB)
            u = lerp(d.uA if k == 0 else d.uB)
            idx = d.idxA if k == 0 else d.idxB
            alert = d.alertA if k == 0 else d.alertB
            only = d.onlyA if k == 0 else d.onlyB
            pos = panel["pos"]
            draw_edges(cv, th, pos, W, idx=idx, alert=alert if alert.any() else None,
                       width=3.2, alpha=0.7)
            cols_u = np.tile(hex_rgb(th.node_mid), (d.n, 1))
            cols_u[idx] = fiedler_colours(th, u)
            present = np.zeros(d.n, bool)
            present[idx] = True
            ring = np.zeros(d.n, bool)
            ring[idx[only]] = True
            draw_nodes(cv, th, pos, cols_u, self.r_node, ring=ring, ghost=~present)
            colours, alphas = [], []
            for uidx in range(d.n):
                if not present[uidx]:
                    colours.append(th.muted)
                    alphas.append(0.45)
                elif ring[uidx]:
                    colours.append(th.alert)
                    alphas.append(1.0)
                else:
                    colours.append(th.ink2)
                    alphas.append(1.0)
            draw_circle_labels(cv, th, pos, panel["centre"], self.R, self.r_node,
                               d.union_tokens, self.label_fs, colours=colours, alphas=alphas,
                               radial=self.radial)
            # caption and readouts
            x0, y0 = panel["x0"], panel["y0"]
            name = d.labels[k]
            cv.text(x0, y0 + self.cap_fs, name, self.cap_fs, th.ink if k == 0 else th.alert,
                    weight="semibold", z=50)
            lam = lerp(d.lamA if k == 0 else d.lamB)
            hf = lerp(d.hferA if k == 0 else d.hferB)
            parts = [f"λ₂ {lam:.3f}"]
            if np.isfinite(hf):
                parts.append(f"HFER {hf:.3f}")
            if k == 1 and np.isfinite(d.span_share).any():
                parts.append(f"{d.labels[1]} tokens hold {lerp(d.span_share):.0%} of the "
                             "edge weight" if not cv.compact else f"span {lerp(d.span_share):.0%}")
            cv.text(x0 + panel["w"], y0 + self.cap_fs, "  ·  ".join(parts), cv.fs(13, 10),
                    th.muted, ha="right", z=50)
        if self.traj_h > 0:
            self._trajectories(p)
        self.header(self.title, self.subtitle)
        self.counter(str(lay), "layer")
        self.progress(p, d.layers, color=th.alert)

    def _trajectories(self, p: float):
        cv, th, d = self.cv, self.th, self.data
        gap = cv.px(36, 14)
        w = (cv.W - 2 * self.margin - gap) / 2
        y_top = self.bottom - self.traj_h + cv.px(26, 14)
        h = self.traj_h - cv.px(40, 22)
        L = len(d.layers)
        series = [("Fiedler value λ₂", d.lamA, d.lamB), ("HFER", d.hferA, d.hferB)]
        fs = cv.fs(12.5, 10)
        for k, (name, a, b) in enumerate(series):
            x0 = self.margin + k * (w + gap)
            if not (np.isfinite(a).any() and np.isfinite(b).any()):
                continue
            both = np.concatenate([a[np.isfinite(a)], b[np.isfinite(b)]])
            lo, hi = float(both.min()), float(both.max())
            padv = (hi - lo) * 0.12 or 0.05
            lo, hi = lo - padv, hi + padv
            X = lambda t: x0 + w * np.asarray(t) / max(L - 1, 1)  # noqa: E731
            Y = lambda v: y_top + h - h * (np.asarray(v) - lo) / (hi - lo)  # noqa: E731
            cv.text(x0, y_top - cv.px(8, 4), name, fs, th.ink2, weight="semibold", z=30)
            nw, _ = cv.text_size(name, fs, "semibold")
            lx = x0 + nw + cv.px(16, 8)
            for lab, col in ((d.labels[0], th.ink2), (d.labels[1], th.alert)):
                cv.lines([[(lx, y_top - cv.px(12, 7)), (lx + cv.px(16, 10), y_top - cv.px(12, 7))]],
                         [col], [cv.px(2.4, 1.6)], z=30)
                cv.text(lx + cv.px(21, 13), y_top - cv.px(8, 4), lab, fs, th.muted, z=30)
                lx += cv.px(21, 13) + cv.text_size(lab, fs)[0] + cv.px(14, 8)
            cv.lines([[(x0, y_top + h), (x0 + w, y_top + h)]], [th.rule], [cv.px(1, 0.7)], z=29)
            tf = cv.fs(11, 9.5)
            cv.text(x0 + w, y_top + h + tf * 1.3, f"{hi - padv:.2f} max · {lo + padv:.2f} min",
                    tf, th.muted, ha="right", z=30)
            xs = X(np.arange(L))
            for v, col, lw in ((a, th.ink2, 2.0), (b, th.alert, 2.2)):
                pts = np.stack([xs, Y(v)], 1)
                cv.lines([pts], [rgba(hex_rgb(col), 0.28)[0]], [cv.px(lw * 0.8, 1)], z=30)
                k_ = int(np.floor(p))
                e = p - k_
                done = pts[:k_ + 1]
                if k_ + 1 < L:
                    done = np.concatenate([done, [pts[k_] + e * (pts[k_ + 1] - pts[k_])]])
                if len(done) > 1:
                    cv.lines([done], [col], [cv.px(lw, 1.3)], z=31)
                cv.circles([done[-1]], cv.px(4.5, 3), [col], edge=th.bg, lw=cv.px(1.5, 1), z=32)
            cx = float(X(p))
            cv.lines([[(cx, y_top), (cx, y_top + h)]], [rgba(hex_rgb(th.rule), 0.9)[0]],
                     [cv.px(1, 0.7)], z=29)


def compare_html(cache, path, theme, *, other=None, texts=(0, 1), layers=None,
                 labels=("clean", "injected"), plotlyjs: str = "cdn", title=None, **_ignored):
    """Interactive version: both graphs and the two trajectories, with a
    layer slider; hover a node for its token and Fiedler entry."""
    d = CompareData(cache, other, texts, layers, labels)
    L = len(d.layers)
    pos = circle_positions(d.n, 0.0, 0.0, 1.0)
    pos[:, 1] *= -1                                    # plotly's y points up

    def panel(k, li):
        W = d.WA[li] if k == 0 else d.WB[li]
        u = d.uA[li] if k == 0 else d.uB[li]
        idx = d.idxA if k == 0 else d.idxB
        only = d.onlyA if k == 0 else d.onlyB
        toks = d.tokA if k == 0 else d.tokB
        ax = "" if k == 0 else "2"
        n = W.shape[0]
        iu, ju = np.triu_indices(n, 1)
        w = W[iu, ju]
        rel = w / (w.max() or 1.0)
        traces = []
        for lo, hi, a, width, hot in ((0.08, 0.35, 0.22, 1.2, False), (0.35, 1.01, 0.6, 2.6, False),
                                      (0.08, 1.01, 0.75, 2.2, True)):
            sel = (rel > lo) & (rel <= hi)
            touch = only[iu] | only[ju]
            sel &= touch if hot else ~touch
            xs, ys = [], []
            for a_, b_ in zip(iu[sel], ju[sel]):
                xs += [pos[idx[a_], 0], pos[idx[b_], 0], None]
                ys += [pos[idx[a_], 1], pos[idx[b_], 1], None]
            col = theme.alert if hot else theme.edge
            r, g, b = (int(v * 255) for v in hex_rgb(col))
            traces.append({"type": "scatter", "mode": "lines", "x": _r(xs), "y": _r(ys),
                           "xaxis": "x" + ax, "yaxis": "y" + ax, "hoverinfo": "skip",
                           "line": {"color": f"rgba({r},{g},{b},{a})", "width": width}})
        cols = fiedler_colours(theme, u)
        hover = [f"<b>{t}</b><br>Fiedler {u[q]:+.3f}" for q, t in enumerate(toks)]
        traces.append({"type": "scatter", "mode": "markers+text", "x": r4(pos[idx, 0]),
                       "y": r4(pos[idx, 1]), "text": toks, "textposition": [
                           "middle right" if pos[q, 0] > 0.2 else "middle left" if pos[q, 0] < -0.2
                           else ("top center" if pos[q, 1] > 0 else "bottom center") for q in idx],
                       "hovertext": hover, "hoverinfo": "text", "xaxis": "x" + ax,
                       "yaxis": "y" + ax,
                       "textfont": {"color": [theme.alert if o else theme.ink2 for o in only],
                                    "size": 12},
                       "marker": {"size": 11, "color": ["rgb({},{},{})".format(
                           *(int(round(v * 255)) for v in c)) for c in cols],
                           "line": {"color": [theme.alert if o else theme.node_line for o in only],
                                    "width": [3 if o else 1 for o in only]}}})
        return traces

    def cursor(li):
        out = []
        for ax, a, b in (("3", d.lamA, d.lamB), ("4", d.hferA, d.hferB)):
            out.append({"type": "scatter", "mode": "markers", "x": [d.layers[li]] * 2,
                        "y": [_f(a[li]), _f(b[li])], "xaxis": "x" + ax, "yaxis": "y" + ax,
                        "marker": {"size": 9, "color": [theme.ink2, theme.alert]},
                        "hoverinfo": "skip"})
        return out

    static = []
    for ax, a, b in (("3", d.lamA, d.lamB), ("4", d.hferA, d.hferB)):
        for v, col, name in ((a, theme.ink2, d.labels[0]), (b, theme.alert, d.labels[1])):
            static.append({"type": "scatter", "mode": "lines", "x": list(d.layers),
                           "y": [_f(x) for x in v], "xaxis": "x" + ax, "yaxis": "y" + ax,
                           "name": name, "line": {"color": col, "width": 2},
                           "hovertemplate": name + " · layer %{x}: %{y:.3f}<extra></extra>"})
    first = panel(0, 0) + panel(1, 0)
    n_dyn = len(first)
    data = first + cursor(0) + static
    dyn_idx = list(range(n_dyn + 2))
    frames = [{"name": f"L{lay}", "data": panel(0, li) + panel(1, li) + cursor(li),
               "traces": dyn_idx} for li, lay in enumerate(d.layers)]
    menus, sliders = plotly_player([f["name"] for f in frames], [str(x) for x in d.layers],
                                   theme, frame_ms=350, redraw=False)
    hidden = {"visible": False, "range": [-1.45, 1.45], "fixedrange": True}
    axd = {"color": theme.muted, "gridcolor": theme.faint, "zeroline": False}
    layout = plotly_layout(
        theme, title or f"{d.labels[0].capitalize()} vs {d.labels[1]}",
        f"{model_label(d.A)} · shared tokens in the same place; {d.labels[1]}-only tokens and "
        "their edges in vermillion",
        xaxis=dict(hidden, domain=[0, 0.48]), yaxis=dict(hidden, domain=[0.36, 1],
                                                         scaleanchor="x"),
        xaxis2=dict(hidden, domain=[0.52, 1]), yaxis2=dict(hidden, domain=[0.36, 1],
                                                           scaleanchor="x2"),
        xaxis3=dict(axd, domain=[0, 0.46], anchor="y3", title={"text": "layer"}),
        yaxis3=dict(axd, domain=[0.08, 0.27], anchor="x3", title={"text": "λ₂"}),
        xaxis4=dict(axd, domain=[0.54, 1], anchor="y4", title={"text": "layer"}),
        yaxis4=dict(axd, domain=[0.08, 0.27], anchor="x4", title={"text": "HFER"}),
        updatemenus=menus, sliders=sliders,
        annotations=[{"text": f"<b>{d.labels[0]}</b>", "x": 0.0, "y": 1.0, "xref": "paper",
                      "yref": "paper", "showarrow": False, "xanchor": "left",
                      "font": {"color": theme.ink, "size": 15}},
                     {"text": f"<b>{d.labels[1]}</b>", "x": 0.52, "y": 1.0, "xref": "paper",
                      "yref": "paper", "showarrow": False, "xanchor": "left",
                      "font": {"color": theme.alert, "size": 15}}])
    layout["margin"]["b"] = 110
    for s in layout["sliders"]:
        s["y"] = -0.04
    for m in layout["updatemenus"]:
        m["y"] = -0.04
    fig = {"data": data, "layout": layout, "frames": frames}
    return write_plotly_html(path, fig, f"compare · {model_label(d.A)}", theme, plotlyjs)


def _r(xs):
    return [None if v is None else round(float(v), 4) for v in xs]


def _f(v):
    v = float(v)
    return round(v, 5) if np.isfinite(v) else None
