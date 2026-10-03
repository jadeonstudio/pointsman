# Isolated Clef-Flash DEV experiment

**Verdict: both local candidates are NO-GO.** The [frozen aggregate](evidence.json)
records 4-bit/8-bit oracle agreement of 52.0%/51.2%, rule/order pairs 0/9 for both,
and Korean 14/41 versus 15/41. Eight Noul rows have an underspecified-priority
caveat; the nine changed-rule Choice pairs are unambiguous and unaffected.
These are one-family DEV results for converted models, not publisher BF16 or
Jev results. No checkpoint was promoted and no runtime provider was added.

Frozen before inference: dataset `275d73372d3ed2d57320072fa6b6c682cf272832c295a0d9eff261e1e1c4c935`, DEV only (125 rows). Baseline strict accuracy 53.6%, macro F1 0.329167, rule checks 4/9, Noul 5/20, Korean 22/41. Hypothesis: a pretrained schema-conditioned 9B backbone may improve these failed boundaries. Improvements are synthetic DEV evidence, not adoption or operational utility.

Public upstream: `Cloudflare/clef-flash@17f0b0ad64efb65d273590632833508766b2aae6`. Private environment and artifacts live under `~/.local/share/pointsman-research/2026-10-03/clef/`. The active Laya environment and Pointsman providers/hooks/policy are untouched. No paid calls or training.

`evaluate.py preflight` checks BF16 Metal DLPack and the unchanged official PyTorch joint head before the backbone download. `prepare` converts the official text backbone through MLX-LM with 4-bit/group-64 affine linear weights; input/output embeddings and joint head stay BF16. Quantization is lossy. `predict` preserves the existing wire-request wrapper, admits only complete text records at most 2048 tokens, uses B1/no padding/no autoregressive generation, and emits sample-ID prediction rows for the existing general-decisions evaluator.

At most three causally justified preparation attempts; unsupported operations stop this path rather than triggering a framework port. Required checks: pinned input hashes, complete input admission, option identity/order, same-hidden-input head bridge equality, finite probability mass, and actual elapsed time/peak memory. Never infer calibration/test rows or read raw records into the chat.

## 2026-10-03 execution

First preparation attempt passed: MLX 0.32.3, mlx-lm 0.32.0, Torch 2.14.0, Transformers 5.17.0, Safetensors 0.8.0. Anonymous `hf download` and `hf cache verify` checked all 17 pinned upstream files. The actual official head loaded strictly; BF16 Metal DLPack produced finite logits with maximum same-hidden-input difference 0.0 against copied Torch inputs.

Conversion took 26.514 seconds, with MLX peak 7,962,256,392 bytes and RSS peak 3,682,664,448 bytes. One DEV inference pass served all 125 rows, with full encoded lengths 284–356 tokens, 138.863 seconds of loop time, MLX peak 8,602,091,386 bytes, and RSS peak 2,009,612,288 bytes. First call was 4476.579 ms; the remaining 124 calls had median 1082.679 ms and p95 1253.997 ms. Later filesystem timing readback established that the Node CPU gate overlapped the initial 40.4 seconds; Python finished before the prediction loop, and other host load was uncontrolled. Model load time was not separately recorded; these numbers are not a matched-baseline speed comparison. The frozen manifest's stronger Node/Python overlap wording is corrected separately in `artifacts/erratum.json`, without changing its metrics or identity.

The derived checkpoint is `ef744cdfbba595a68a082368f047d6472a6e5ef576c97c074fb15d3291001b97`; prediction-file SHA256 is `2d3e78a08f0425c2d62c7605b4f6786aaaa03fe463ba329519bcae3b2c46e442`. Private `artifacts/preflight.json`, `preparation.json`, `predictions.jsonl`, and `manifest.json` preserve precision, resource limits, runtime, per-row shared-wire identity, and selected upstream asset hashes. The original checkpoint omits some tokenizer/configuration assets; `artifacts/full-upstream-hashes.json` supplements them with a later complete pinned-file readback, without redefining the checkpoint or proving continuous mid-run immutability. Quality scoring belongs to the existing `training/general-decisions/evaluate.mjs`; no quality or adoption conclusion is made here.

Reproduce in a fresh isolated directory with Python 3.12.8. The private directory must contain `upstream/` with all 17 official files; no `special_tokens_map.json` is present in this revision. The measured dependency versions and anonymous download layout are:

```sh
# Choose a fresh CLEF_ROOT; preserve the measured 2026-10-03 directory.
CLEF_ROOT="$HOME/.local/share/pointsman-research/<new-run>/clef"
CLEF_PYTHON="<path-to-python-3.12.8>"
"$CLEF_PYTHON" -m venv "$CLEF_ROOT/.venv"
"$CLEF_ROOT/.venv/bin/python" -m pip install 'mlx==0.32.3' 'mlx-lm==0.32.0' 'torch==2.14.0' 'transformers==5.17.0' 'safetensors==0.8.0' 'huggingface-hub==1.33.0'
env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME="$CLEF_ROOT/hf" "$CLEF_ROOT/.venv/bin/hf" download Cloudflare/clef-flash --revision 17f0b0ad64efb65d273590632833508766b2aae6 --exclude 'model-*.safetensors' --local-dir "$CLEF_ROOT/upstream" --max-workers 2
"$CLEF_ROOT/.venv/bin/python" training/clef-local/evaluate.py preflight --root "$CLEF_ROOT"
# Download the backbone only after the preflight succeeds.
env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME="$CLEF_ROOT/hf" "$CLEF_ROOT/.venv/bin/hf" download Cloudflare/clef-flash --revision 17f0b0ad64efb65d273590632833508766b2aae6 --include 'model-*.safetensors' --local-dir "$CLEF_ROOT/upstream" --max-workers 2
env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME="$CLEF_ROOT/hf" "$CLEF_ROOT/.venv/bin/hf" cache verify Cloudflare/clef-flash --revision 17f0b0ad64efb65d273590632833508766b2aae6 --local-dir "$CLEF_ROOT/upstream" --fail-on-missing-files
```

After that preparation, run from the repository root using the frozen DEV dataset home:

```sh
"$CLEF_ROOT/.venv/bin/python" training/clef-local/evaluate.py prepare --root "$CLEF_ROOT"
"$CLEF_ROOT/.venv/bin/python" training/clef-local/evaluate.py predict --root "$CLEF_ROOT" --home "$HOME/.local/share/pointsman-research/2026-10-03"
python3 -m unittest discover -s tests -p test_clef_local.py
```

`prepare` refuses an existing `mlx4/` directory; `predict` refuses an existing `artifacts/predictions.jsonl`. JSON receipts, including `preflight.json`, can overwrite their own filenames. Preserve the frozen run and use a fresh experiment directory for a separately authorized rerun. Set `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`, a private `HF_HOME`, and unset inherited HF token variables for downloads; inference uses local files only. MLX's per-process allocation limit is 14 GiB and cache limit 256 MiB. No OS/global memory setting was changed.

## Frozen 8-bit precision ablation

The measured 4-bit evaluator was preserved byte-for-byte at private `artifacts/evaluate-4bit-executed.py` (SHA256 `235ac33ec0d02523868da01623a348083feaaa75e7e14c20ae3b8ed1e043c4c7`) before adding the optional `--bits 4|8` argument. Default remains 4; 8 uses separate `mlx8/` and `artifacts8/` paths and derives a distinct model/checkpoint identity. `prepare --bits 8` reuses the original pinned BF16 upstream weights and verified head preflight, not dequantized 4-bit weights; `predict --bits 8` uses that preparation. No repeated preflight or download is needed.

Hypothesis frozen before the 8-bit inference: if 4-bit loss caused the rule/Korean Choice failures, 8-bit should improve the nine unambiguous rule pairs and Korean Choice slice on the exact same original 125 DEV rows, inputs, option order, and BF16 head. Eight Noul rows have a source-oracle ambiguity under separate diagnosis; do not change these inputs/labels within this ablation or treat every original row as fully validated ground truth. Overall gates remain descriptive under that caveat. No stochastic inference/training or model adoption is introduced.

From the actual 4-bit tensor headers, an 8-bit backbone projects to 11,421,508,608 bytes, with unchanged group-64 scale/bias and BF16 embeddings. Adding the observed 4-bit workspace suggests about 11.23 GiB MLX peak, plus the 243.53 MB official head. This is an estimate, not an execution guarantee. One conversion and one DEV pass are authorized under the unchanged 14 GiB MLX limit; OOM/non-finite/budget failure stops the experiment. Absolute stage timestamps and model-load time are recorded for this new run.

The one authorized 8-bit conversion passed in 31.101 seconds with MLX peak 11,421,508,616 bytes and RSS peak 5,964,857,344 bytes. One DEV pass served all 125 rows in 158.303 seconds of loop time; model/head load took 3.534 seconds. First call was 10,676.530 ms; remaining-call median was 1207.773 ms and p95 1352.153 ms. MLX peak was 11,862,750,856 bytes and RSS peak 3,132,194,816 bytes. Stage timestamps are `2026-10-03T05:31:10.731117Z`–`2026-10-03T05:34:01.845531Z`; CPU gates had completed per the root, while other host load was uncontrolled.

Direct 4/8 readback confirmed all 125 rows share the same sample IDs/order, shared/full wire hashes, encoded token hashes/lengths, and head option order. The 8-bit checkpoint is `2d78acadca4a2d3865b6c9efd8402d1b2483c1dc6c43a802f81ecc72dcb45e63`; prediction-file SHA256 is `dad2e3b3f7337942598881a7b0658886bb9b7acc3b05284b48927ea111723fd0`. Private `artifacts8/` holds the frozen results; the GPU process exited and no further inference was run. These execution checks leave quality scoring to the original evaluator and preserve the source-oracle caveat.
