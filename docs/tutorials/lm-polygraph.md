# LM-Polygraph bridge

[LM-Polygraph](https://github.com/IINemo/lm-polygraph) (Fadeeva et al.,
2023; MIT licence) is a benchmark of uncertainty estimators for language
models: datasets, generation, quality metrics and several dozen estimators,
all scored the same way. `spectral_trust.integrations.lm_polygraph` turns
any attention metric of this library into one of its estimators, so that a
spectral score can be compared with LM-Polygraph's baselines on its own
benchmark.

```bash
pip install "spectral_trust[polygraph]"      # lm-polygraph >= 0.7, Python >= 3.10
```

The core package never imports lm-polygraph; only
`spectral_trust.integrations.lm_polygraph` does, and it raises an
`ImportError` that names the extra when lm-polygraph is missing.

## How it plugs in

LM-Polygraph computes statistics once per batch with its stat calculators
and passes them to every estimator; an estimator names the statistics it
needs and returns one value per item, higher meaning more uncertain.
`SpectralTrustEstimator` asks for two of them:

- `forwardpass_attention_weights`: a forward pass over the prompt followed
  by the greedy generation, `[layers, heads, S, S]` per item (NaN-padded
  in a batch);
- `greedy_tokens`, to locate the generated tokens at the end.

For each item it runs `spectral_trust.compute_spectra` on the attention,
reduces the per-head values over heads, then over layers, and multiplies by
`sign`.

```python
from spectral_trust.integrations.lm_polygraph import SpectralTrustEstimator

est = SpectralTrustEstimator(
    "operator.participation_ratio",   # any attention metric, namespaced
    layers=None,                      # every layer; or an index or a list
    head_aggregation="mean",          # mean, median, max, min or sum
    layer_aggregation="mean",
    sign=-1,                          # a lower ratio scores as more uncertain
    span="all",                       # "all", "prompt" or "generated"
)
```

The metric can be any attention metric of the library:
`operator.participation_ratio`, `operator.contraction`,
`graph.fiedler_value`, `graph.forman_curvature_mean`, `attn.hfer`, and so
on, with the graph and operator options of a `GSPConfig` passed as
`config=`. Hybrid metrics (which read hidden states) are refused, since the
statistic holds attention only, and so are value-weighted edges
(`edge_weights`, `operator_edges`). `span="generated"` measures the subgraph
of the generated tokens, the analogue of `token_span`; an empty span scores
NaN, which LM-Polygraph leaves out of its metrics. `str(est)` names the
estimator in LM-Polygraph's tables after its arguments, or `name=`.
`est.score(attention, n_generated)` scores one item without LM-Polygraph's
manager.

The direction of a score is a choice to make before looking at the
results: LM-Polygraph reads higher as more uncertain, and `sign` says which
way the metric is expected to move.

## On the benchmark

A run with LM-Polygraph's own manager, stat calculators and metrics, here
on a few items of its CoQA dataset (`LM-Polygraph/coqa`, configuration
`continuation`, columns `input` and `output`):

```python
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.utils.generation_parameters import GenerationParameters
from lm_polygraph.utils.manager import UEManager
from lm_polygraph.utils.dataset import Dataset
from lm_polygraph.utils.builder_enviroment_stat_calculator import BuilderEnvironmentStatCalculator
from lm_polygraph.defaults.register_default_stat_calculators import register_default_stat_calculators
from lm_polygraph.estimators import MaximumSequenceProbability
from lm_polygraph.generation_metrics import RougeMetric
from lm_polygraph.ue_metrics import PredictionRejectionArea
from spectral_trust.integrations.lm_polygraph import SpectralTrustEstimator

df = pd.read_parquet("hf://datasets/LM-Polygraph/coqa/continuation/test-00000-of-00001.parquet")
x, y = df["input"].tolist()[:6], df["output"].tolist()[:6]

tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
hf = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager").eval()
model = WhiteboxModel(hf, tok, "gpt2", generation_parameters=GenerationParameters(max_new_tokens=12))

estimators = [SpectralTrustEstimator("operator.participation_ratio", sign=-1),
              MaximumSequenceProbability()]
man = UEManager(Dataset(x, y, batch_size=1), model, estimators,
                builder_env_stat_calc=BuilderEnvironmentStatCalculator(model),
                available_stat_calculators=register_default_stat_calculators("Whitebox"),
                generation_metrics=[RougeMetric("rougeL")],
                ue_metrics=[PredictionRejectionArea()], processors=[],
                ignore_exceptions=False, verbose=False, max_new_tokens=12)
man()
man.estimations[("sequence", str(estimators[0]))]    # one score per item
man.metrics                                          # PRR of every estimator
```

Use `batch_size=1` for attention estimators. LM-Polygraph's forward-pass
statistic concatenates each item's tokenised prompt, padded to the batch,
with its generation and runs it without an attention mask, so in a batch of
prompts of different lengths the shorter ones are measured with pad tokens
in the graph (this holds for its own attention baselines too).

Tested with lm-polygraph 0.7.0, torch 2.15.0 (CPU), transformers 5.23.0 and
Python 3.12: on GPT-2 and six CoQA items, four estimators of this library
(operator participation ratio, contraction on the generated tokens, Fiedler
value, mean Forman curvature) and the maximum sequence probability baseline
all return finite scores, and the manager computes their PRR. Six items
are far too few for the PRR to mean anything. `WhiteboxModel.from_pretrained`
is deprecated in 0.7.0; the model is built from a loaded model, with
`attn_implementation="eager"` so that attention weights are returned.
`tests/test_lm_polygraph_bridge.py` runs the same manager on a tiny random
GPT-2 wherever lm-polygraph is installed.
