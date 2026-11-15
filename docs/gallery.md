# Animated gallery

New in 0.4.2: one call writes a video, a GIF or an interactive page of a
`SpectralCache`. The file suffix picks the format. The videos on this page
were rendered by `docs/make_gallery.py` on Qwen3-0.6B.

```python
import spectral_trust as st

model = st.SpectralModel.from_pretrained("Qwen/Qwen3-0.6B")
cache = model.run_with_spectra("The cat sat on the mat because it was warm.")

cache.animate("waterfall", "waterfall.mp4")              # or st.viz.animate(cache, ...)
```

| Suffix | Output | Default size and rate |
| --- | --- | --- |
| `.mp4` | H.264 video | 1280 × 720, 30 fps |
| `.webm` | VP9 video | 1280 × 720, 30 fps |
| `.gif` | looping GIF, sized for phones | 640 × 360, 15 fps |
| `.html` | interactive plotly.js page | fills the window |

Video and GIF need the `viz` extra (`pip install "spectral_trust[viz]"`,
which brings imageio and its ffmpeg binary). An HTML page embeds plotly.js
when the `plotly` package is installed, so it is one file that works offline;
otherwise it loads a pinned copy from the CDN (`plotlyjs="cdn"` asks for that).
Every call takes `theme="light"` or `"dark"`, `size`, `fps`, `duration`, `dpi`,
`layers` and `wordmark=True` (a small "spectral-trust" in a corner, off by
default). Output is deterministic for a fixed `seed`. Rendering runs on the CPU,
and ffmpeg uses two threads (`threads=`).

## flow

<video controls muted loop playsinline preload="none" width="100%" poster="https://vcnoel.github.io/media/spectral-trust/0.4.2/flow-poster.png" aria-label="flow view of Qwen3-0.6B reading an email with an injected instruction">
  <source src="https://vcnoel.github.io/media/spectral-trust/0.4.2/flow.webm" type="video/webm">
  <source src="https://vcnoel.github.io/media/spectral-trust/0.4.2/flow.mp4" type="video/mp4">
</video>

[Open the interactive page](https://vcnoel.github.io/media/spectral-trust/0.4.2/flow.html) (orbit and hover).

```python
pair = model.run_with_spectra([clean, injected])
pair.animate("flow", "flow.mp4", text=1, other=pair.text(0), theme="dark")
```

Each token is a tube through depth. Along the tube is the layer, from 0 to the
last. Across it are the token's spectral coordinates at that layer: the Fiedler
vector upright, with its sign matched from layer to layer, and eigenvector 3 in
depth. A Catmull-Rom spline joins the layers. The colour is the token's side
of the Fiedler split at that layer, and the thickness is the attention the
token receives, so a sink becomes a trunk. The sentence braids into two
bundles where the split forms and crosses over where context mixes. With
`other=` (the prompt without the injection), the tokens only this prompt has
are outlined in vermillion, and they travel as their own strand. A light,
display-only pull toward each side's mean (`bundle=0.45`, 0 for the raw
coordinates) makes the bundles easy to see. `axes="procrustes"` uses
Procrustes-aligned eigenvectors 2 and 3 instead.

The camera starts at layer 0, flies along the depth axis through the layers,
then pulls back and turns to show the whole braid. Labels mark only the ends
of a few key tokens. The scene is three.js (r160, shipped with the package):
the `.html` page plays the flight once, then can be orbited and hovered.
Videos are captured frame by frame from the same page in headless Chrome,
Edge or Chromium, so they look the same as the page. Without a browser,
`engine="matplotlib"` draws a simpler version of the same scene and camera.
`SPECTRAL_TRUST_BROWSER` points at a browser that is not found on its own.

## embedding3d

```python
cache.animate("embedding3d", "embedding3d.mp4", theme="dark")
```

Each token sits at its coordinates in Laplacian eigenvectors 2 to 4 of the
layer's head-averaged attention graph. Its colour is the side of the Fiedler
split, blue or orange. The cloud morphs from layer to layer under a slow camera
orbit. Successive layers are aligned by orthogonal Procrustes after their signs
are fixed, so the motion shows how the geometry changes rather than the
arbitrary signs of the eigenvectors. In a dense cluster the label of the
farther token gives way rather than touch another. `floor=True` adds drop lines
and shadows on a floor under the cloud. The `.html` version can be rotated and
hovered, and its slider moves through the layers.

## waterfall

<video controls muted loop playsinline preload="none" width="100%" poster="https://vcnoel.github.io/media/spectral-trust/0.4.2/waterfall-poster.png" aria-label="waterfall of graph-frequency energy by layer, Qwen3-0.6B">
  <source src="https://vcnoel.github.io/media/spectral-trust/0.4.2/waterfall.webm" type="video/webm">
  <source src="https://vcnoel.github.io/media/spectral-trust/0.4.2/waterfall.mp4" type="video/mp4">
</video>

```python
cache.animate("waterfall", "waterfall.mp4")
```

The residual-stream energy of each graph-frequency mode (`spectral_masses`,
ascending eigenvalue) is drawn as a share of the layer's total. Each layer is a
ridge, and the ridges are built front to back through depth. The HFER band (the
top modes) is shaded orange, and a small inset traces HFER by layer.
`scale="sqrt"` or `"log"` brings out the small modes.

## compare

<video controls muted loop playsinline preload="none" width="100%" poster="https://vcnoel.github.io/media/spectral-trust/0.4.2/compare-poster.png" aria-label="clean and injected prompts compared layer by layer, Qwen3-0.6B">
  <source src="https://vcnoel.github.io/media/spectral-trust/0.4.2/compare.webm" type="video/webm">
  <source src="https://vcnoel.github.io/media/spectral-trust/0.4.2/compare.mp4" type="video/mp4">
</video>

```python
pair = model.run_with_spectra([clean, injected])
pair.animate("compare", "compare.mp4", labels=("clean", "injected"))
# or: st.viz.animate(clean_cache, "compare", "compare.mp4", other=injected_cache)
```

This shows two prompts side by side, layer by layer. Their tokens are matched,
so a shared token sits at the same place in both graphs. Tokens that only the
second prompt has, such as an injected instruction, open a gap in the first
graph. In the second graph they and their edges are drawn in vermillion, with
the share of the edge weight they hold. Below, the Fiedler value and HFER of
both prompts are traced through depth.

## graph_movie

<video controls muted loop playsinline preload="none" width="100%" poster="https://vcnoel.github.io/media/spectral-trust/0.4.2/graph_movie-poster.png" aria-label="token graph of each layer of Qwen3-0.6B">
  <source src="https://vcnoel.github.io/media/spectral-trust/0.4.2/graph_movie.webm" type="video/webm">
  <source src="https://vcnoel.github.io/media/spectral-trust/0.4.2/graph_movie.mp4" type="video/mp4">
</video>

```python
cache.animate("graph_movie", "graph.mp4", layout="circle")   # or "force", "spectral"
```

This is the token graph of each layer. Edges are drawn by their normalised
weight, and nodes are coloured by the Fiedler vector on the library's
blue-white-orange scale. The edges draw in, weakest first, before the graph
morphs through the layers.

## head_atlas

```python
cache.animate("head_atlas", "heads.mp4", metric="fiedler_value")
cache.animate("head_atlas", "heads.html", layer=12)
```

Every head of a layer appears as a small multiple, sorted by a per-head metric
(any `attn.*` or `resid.*` name). The sweep over layers morphs each head in
place, then crossfades to the new order. The `.html` version shows each head's
attention pattern as a heatmap, with token names on hover.

## Rebuilding the examples

`python docs/make_gallery.py` renders every example on Qwen3-0.6B into
`docs/gallery_out/` (not tracked). The script uses the CPU with two threads and
renders one file at a time. It writes `manifest.json` with the render time and
size of each file.
