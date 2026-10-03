# training/laya-kit -- local Laya training and evaluation

The kit runs explicitly authorized local training on Apple Silicon MPS or CPU.
It never starts online learning, automatically promotes a checkpoint, or enables
provider modes. The current execution scope uses existing local hardware and
subscriptions; the optional CUDA/Kaggle path is not part of that local run.
Registration, evaluation and promotion remain separate operator actions within
the owner's approved scope. Preserve the historical d6 route baseline and its
provenance; write every new candidate to a separate output directory.

Pinned official reference:
- Repository commit: `NandhaKishorM/laya@42626c348753fbb17572a813127df2278a1ec527`
- Notebook: `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`
- License: the `LICENSE` at that commit is Apache-2.0. There is no separate
  `NOTICE` file at that same commit (confirmed by browsing the tree).
- `laya`'s own version: `pyproject.toml` at that commit declares `version = "0.3.4"`.

## File layout

| File | Role |
|---|---|
| `check_export.py` | standard-library-only script that inspects an export folder (`train/dev/calibration/test.jsonl`, metadata and manifest; legacy three-way exports also accepted) without training |
| `train_from_export.py` | local or DDP full-encoder training from an export; supervised CE by default |
| `requirements.lock` | the package list the notebook installs, pinned to the extent verifiable |
| `NOTICE` | upstream Apache-2.0 notice |

## Local runbook

Version 3 exports (`laya-typed-decisions-json-v3`) contain four disjoint roles:

| Split | Allowed use |
|---|---|
| `train.jsonl` | Train encoder and dynamic scoring parameters |
| `dev.jsonl` | Select epochs, methods and hyperparameters |
| `calibration.jsonl` | Fit final temperatures and qualification gates |
| `test.jsonl` | Evaluate a frozen candidate under the sealed evaluation specification |

`metadata.jsonl` binds each exported row to its sample/group/source, rights and
oracle/provenance identities without adding those labels to the model input.
`manifest.json` binds per-file counts/hashes, split roles and group/provenance
fingerprints. Keep both files with the data. Source/template/semantic families,
translations, paraphrases and counterfactual siblings stay together. Unknown
rights in historical captures do not become permitted independent gold just
because they are re-exported.

1. **Prepare and validate the export.** Independent corpus generation returns a
   dataset version. Existing captured datasets may request
   `pointsman dataset build --split-version 2`; old default builds and distillation
   datasets retain their legacy format. Export the chosen immutable version:
   ```sh
   pointsman dataset export --version <hash> --format laya
   python3 training/laya-kit/check_export.py --export-dir <export-directory> \
       --dataset-manifest <dataset-manifest.json>
   ```
   Stop on invalid split, lineage, source rights or hash identity. Legacy v2
   exports (`train/calibration/test`) remain readable, but have no independent
   dev set: use fixed hyperparameters and `--no-select-best-epoch`, or prepare
   genuinely four-way data. Do not use calibration to choose epochs.

2. **Check local token admission before training.** Use the installed local
   runtime and an existing permitted multilingual checkpoint. Supply the actual
   checkpoint directory containing `rl_agent_config.json`, `model.safetensors`,
   `encoder/` and `tokenizer/`; add `--model-subdir multilingual` only when
   `--model-dir` names a snapshot root above that directory.
   ```sh
   /Users/jangjiyong/.local/share/laya/.venv/bin/python training/laya-kit/train_from_export.py \
       --export-dir <export-directory> \
       --model-dir /Users/jangjiyong/.local/share/laya/models/multilingual/multilingual \
       --output-dir <new-output-directory> \
       --local --device mps --epochs 1 --batch-size 2 --grad-accum 16 \
       --method supervised --input-fit lossless --dry-run
   ```
   Local mode reads JSONL with the standard library and does not require
   `datasets`; dry-run works with the installed torch/Laya/tokenizer stack.
   It validates the export and actual tokenizer admission, writes preparation
   caches, then stops before training. It needs the model/tokenizer runtime even
   though `check_export.py` alone is standard-library-only. Omitted `--model-dir`
   triggers a download; use the existing local path for the authorized local run.
   `--input-fit lossless` is the default and rejects required-evidence loss.
   Explicit `--input-fit task-head` uses the inference worker's task-prefix fit
   and must match registration's `--input-fit task-head`. Admission uses the
   effective saved model lengths (the notebook-derived kit uses 1024/256), not
   an assumed text length. The default maximum dropped fraction is 2%; correct
   the source/serialization when admission fails rather than silently dropping
   evidence, increasing heads or forcing `--allow-truncation`.

3. **Run the fixed local candidate.** After admission passes, rerun the same
   command without `--dry-run`; keep the data, model and arguments fixed.
   `--method supervised` is the default masked distributional cross-entropy
   objective for Choice, two-option Noul and Score. It trains the full encoder
   and scoring parameters. `--method rlcd-grpo` explicitly selects the preserved
   notebook RLCD/GRPO+CE experiment; it is not the supervised baseline.
   Dev chooses the best epoch (`--select-best-epoch`, default on), then
   calibration fits temperatures on those selected weights. Test inference is
   skipped by default. `--evaluate-sealed-test` explicitly evaluates the final
   saved candidate; freeze the evaluation specification first and keep test
   results out of candidate selection and failure mining. `--max-steps N`
   limits micro-batches, not optimizer updates or full epochs; a bounded smoke
   run is not evidence for full-epoch quality. Resume details are below.

4. **Register and collect predictions.** Registration copies and fingerprints
   the candidate; it does not activate or promote it. Match the training fit:
   ```sh
   pointsman laya register --checkpoint <absolute-output-directory> \
       --python /Users/jangjiyong/.local/share/laya/.venv/bin/python \
       --device mps --precision fp32 --input-fit lossless
   pointsman laya predictions --candidate <candidate-hash> --dataset <hash> --split dev
   pointsman laya predictions --candidate <candidate-hash> --dataset <hash> \
       --split test --sealed-spec <frozen-specification-sha256>
   ```
   `predictions` defaults to dev and emits a JSON packet with sample/model/
   checkpoint IDs, probabilities, actual elapsed milliseconds, status accounting
   and dataset/split/provenance/runtime/fit fingerprints. Test requires the
   frozen specification hash. Timeout, rejection, unsupported input and ordinary
   runtime errors remain in the envelope; cancellation ends the run.

5. **Qualify, compare and explicitly promote within the approved scope.** A
   copied test holdout is a regression floor, not independent confirmation.
   Prospective confirmation must use a separate disjoint dataset:
   ```sh
   pointsman laya holdout freeze --dataset <hash> --name <regression-id> --role regression_copy
   pointsman laya holdout freeze --dataset <fresh-hash> --name <prospective-id> \
       --role prospective --training-dataset <hash>
   pointsman laya qualify --candidate <candidate-hash> --dataset <hash> --holdout <holdout-id>
   pointsman laya compare --candidate <candidate-hash> --holdout <holdout-id>
   pointsman laya promote --candidate <candidate-hash> --holdout <holdout-id>
   ```
   Qualification validates training, preprocessing and split identity, fits
   gates on calibration and checks frozen evaluation evidence. It does not
   promote automatically. Version 3 qualification refuses repeated sealed-test
   consumption for the same candidate; a newly selected candidate needs the
   applicable fresh-test policy. A candidate is ready only when the required
   generated qualification, active-checkpoint comparison and fixed-regression/
   prospective non-regression gates pass. Failed or missing gates remain
   failed/UNKNOWN. Promotion reuses an existing authorization when it covers
   the action; otherwise that separate operator decision remains pending.

## Optional CUDA/Kaggle path

The pinned notebook reference supports `torchrun`/NCCL on two NVIDIA T4 GPUs.
For a separately authorized run, transfer the whole private export plus
`check_export.py`, `train_from_export.py`, `requirements.lock` and `NOTICE`;
install the pinned requirements on that machine. Supply a permitted model
snapshot and run the trainer without `--local` (use `--model-subdir multilingual`
when the snapshot stores that variant below its root). The same split, method,
fit and sealed-test rules apply. This path can download packages/models and
consume cloud compute, so it is separate from the existing local-hardware scope.
There is no automatic public dataset or checkpoint upload.

## Historical notebook timing (not current pilot estimates)

- The official notebook's cell 9 markdown states "~4 to 6 minutes total" for
  just the `torchrun` DDP training step (cell 10) -- **in minutes**. This
  is historical notebook evidence, not a runtime estimate for this kit's
  current supervised objective, corpus or hardware.
- The total time for the whole notebook (package install, model/data
  download, preprocessing, evaluation, saving outputs included) is not
  separately documented officially -- left as **cannot confirm**. Kaggle's
  free tier has a per-session GPU time limit (which can change by account and
  over time), so check your Kaggle account's current quota directly before
  running.
- Kaggle's free GPU quota itself (weekly hours, T4x2 session limits) is
  Kaggle's own policy and cannot be pinned down by this repo's documentation
  -- **cannot confirm**, re-check the Kaggle console before running.
- Cost: using Kaggle's free GPU tier as-is costs nothing extra. Whether to buy
  Kaggle Pro or extra compute is a matter of the user's own account policy;
  this kit does not require it.

## Local mode and historical MPS measurements (2026-09-23)

- The official procedure is **CUDA DDP only**, using the `torch.distributed`
  NCCL backend and `torchrun --nproc_per_node=2`. NCCL is only for
  communication between NVIDIA GPUs, so this path does not run as-is on
  Apple Silicon's MPS device.
- `train_from_export.py --local` works around this: it reuses the **exact
  same** selected objective (supervised CE by default; RLCD+GRPO+CE only
  with `--method rlcd-grpo`), optimizer (AdamW with separate encoder/head LR),
  cosine LR schedule, epoch count (default 4, changeable via `--epochs` only
  under `--local` -- see the section below), per-device batch size, and seed,
  as a single process without `torchrun`/NCCL/DDP (`TRAIN_DDP_SCRIPT` was
  refactored internally into `run_training_loop()`/`finalize_and_save()` so
  the DDP path (`main_ddp`) and the local path (`main_local`,
  world_size=1/rank=0) call the same functions). It reads the export's
  train/dev/calibration JSONL using only the standard-library `json` module
  (`load_jsonl_rows`), without the `datasets` package (not installed in the
  local laya venv).
- **Example invocation** (multilingual base checkpoint, as used in this repo):
  ```sh
  /Users/jangjiyong/.local/share/laya/.venv/bin/python training/laya-kit/train_from_export.py \
      --export-dir exports/<hash>/laya \
      --model-dir /Users/jangjiyong/.local/share/laya/models/multilingual/multilingual \
      --output-dir <output-directory> \
      --local --device mps --method supervised --input-fit lossless
  ```
  - `--device mps|cpu` (default mps); `--grad-accum N` (default: computed
    automatically from `--batch-size` to keep the DDP recipe's effective
    global batch unchanged -- at the default batch-size of 8, grad-accum is 8,
    so 8*8=64, matching DDP's 8*4*2=64); `--batch-size N` (default 8, lowering
    it on OOM automatically scales grad-accum up proportionally to keep the
    effective batch the same); `--mps-autocast bf16|off` (default off -- MPS
    fp16 autocast was measured to be unstable for training, see the
    regularization-flags section below; bf16 is opt-in); `--max-steps N`
    (smoke-test only, runs exactly that many micro-steps and stops).
  - Lossless admission is the default; task-head fitting is explicit.
    Both paths use the same effective model configuration and tokenizer.
  - Memory safety: logs `torch.mps.driver_allocated_memory()` every epoch
    (`[mem] epoch N mps_driver_allocated_mib=...`; MPS has no real peak
    counter, so this is a best-effort approximation). On OOM (a
    `RuntimeError` containing "out of memory"), training stops with a
    concrete message suggesting you halve `--batch-size` and scale
    `--grad-accum` up proportionally.
  - Outputs: `model.safetensors` (fp16, same as the notebook -- if best-epoch
    selection is on, this is the weights of the selected epoch), `encoder/`,
    `tokenizer/`, `rl_agent_config.json` (including model_name/
    base_model_dir_name/exporter_version, same layout as DDP),
    `training_metadata.json` (`mode:"local"`, `device`, hyperparameters
    including the actual micro_batch/grad_accum/select_best_epoch,
    `local_run` with device/grad_accum/epochs_completed/wall_time_s,
    `epoch_selection`/`selected_epoch` -- see the section below), and a
    local-only `local_training_run.json`.
- **Smoke test measurement (2026-09-23, M4 Pro, multilingual base, synthetic
  export of 12/6/6 rows, `--local --device mps`)**: with synthetic data, 12
  train rows x 3 questions each = 36 training sequences. With `--max-steps 20`
  (20 micro-batches, batch-size 8/grad-accum 8), total time was 11.953 seconds
  -> about **0.60 seconds/micro-step** (the first step is slower, about 1.8
  seconds, due to MPS kernel compilation -- the steady-state rate is after
  that). The checkpoint loaded immediately with
  `laya.agent.Agent(output_dir, device='mps')` and responded normally to a
  route-shaped question (`intent` choice) (`predict()` took 0.64 seconds).
  `check_tokenizer_admission` also passed cleanly (0/54 truncated).
- **Real-scale measurement (2026-09-23, M4 Pro 24GB, multilingual base, real
  distilled data with 7,659 train rows, up to 1,024 tokens)**:
  - `--batch-size 8` (default): the active memory footprint is large enough
    that free memory drops to 21% and swap kicks in (about 18GB), giving
    **about 8.5 seconds/micro-step** (a projected ~9 hours for 4 epochs). The
    smoke test's 0.60 seconds did not reveal this memory bottleneck because
    its synthetic rows were short (MPS unified memory does not show up in
    process RSS).
  - `--batch-size 4` (grad-accum 16, same effective batch of 64): **about
    0.72-0.88 seconds/micro-step**, about 26 minutes for 1 epoch's 1,915
    steps, projected about 1 hour 45 minutes for 4 epochs. This historical
    run used batch-size 4 to avoid the batch-size-8 swap bottleneck.
  - Swap usage was checked with `sysctl vm.swapusage`; speed was checked from
    the timestamps on the log's every-50-steps lines.
- **Projected time for the owner's target data scale (a projection -- superseded
  by the real measurements above)**: per the owner's plan, the real data scale
  is about 3,000 task sentences x 3 questions = **about 9,000 training
  sequences** (assuming no truncation, consistent with the 0% truncation rate
  measured in the smoke test). At the official epochs=4, batch-size=8, the
  micro-batch count is `ceil(9000/8) * 4 = 4,500`. Multiplying by the smoke
  test's steady-state 0.60 seconds/step gives `4,500 * 0.60s ~= 2,700 seconds
  ~= 45 minutes` (**a projection for the training loop alone** -- model
  loading (~26 seconds), calibration temperature fitting and, when explicitly
  requested, test evaluation (historically about 0.6 seconds per case) add time, and MPS throttling
  or thermal effects at the real 9,000-sequence scale were not observed, so
  the actual number may differ). Compared to the official notebook's "~4 to 6
  minutes" (2xT4 DDP, world_size=2), a single local MPS process takes about
  7-11x longer, but costs $0 and consumes no Kaggle GPU quota.
- The pointsman training-data spec's example `providers.json` showing
  `"device": "mps"` refers to the local Laya worker's device setting **at
  inference time**, which is separate from this training kit's `--local`
  training path (though they use the same physical device). Do not confuse
  the two.

## Selecting the best checkpoint per epoch (`--select-best-epoch`, default ON)

At each epoch end, the shared local/DDP callback evaluates only `dev_items.pt`
and logs `[select] epoch N dev_agreement choice=.. score=.. noul=.. mean=..`.
It selects the weights with the best mean per-type argmax agreement. Final
calibration temperature fitting happens after loading those selected weights,
using `calib_items.pt` separately. DDP evaluates on rank 0 and synchronizes all
ranks. Local resumable runs retain the best snapshot on disk.
`--no-select-best-epoch` uses the final epoch; it is required for legacy v2
exports without independent dev. `epoch_selection.json` and
`training_metadata.json` record the selected epoch, per-epoch table and dev role.

Historical 2026-09-23 route/distillation work selected epochs on calibration:
train agreement about 0.99/0.92/0.95 versus held-out test 0.72/0.58/0.60
motivated checkpoint selection. Those measurements describe the old procedure;
they are not independent dev evidence or current supervised-corpus quality.
Current separation/CE/resume fixture checks live in
[`test_laya_kit_splits.py`](../../tests/test_laya_kit_splits.py) and
[`test_laya_kit_resume.py`](../../tests/test_laya_kit_resume.py).

## Regularization flags to mitigate overfitting (`--dropout`, `--rdrop-alpha`, default OFF, `--local` only, 2026-09-24)

- **Background**: in a second training run, overfitting was severe -- train
  subset agreement of 0.98/0.92/0.94 versus held-out 0.81/0.64/0.59. The
  multilingual base's `encoder/config.json` has attention/embedding/mlp
  dropout all set to 0.0, so the encoder trains with no dropout (note: the
  `DecisionModel`'s 2-layer head has always used a hard-coded
  `nn.TransformerEncoderLayer` dropout of 0.1 from laya's own code).
- **`--dropout P`** (0 < P <= 0.5, unset by default = keep the checkpoint's own
  setting): sets the encoder's attention/embedding/mlp dropout to P during
  training. ModernBERT copies this value at module-construction time, and
  makes the attention output dropout an `nn.Identity` when it is 0, so
  changing the setting after construction misses some of it. To handle this,
  `build_model_with_encoder_dropout()` copies the encoder directory to a
  temporary location, changes only the config, builds the model with laya's
  `build_model()`, and right after construction
  `assert_encoder_dropout_applied()` inspects the actual module values (for
  multilingual, aborts unless all 45 Dropout modules + 22 attention modules
  equal P; the result is logged as `[local] encoder dropout applied: ...` and
  recorded in `local_training_run.json`'s `encoder_dropout_check`). **The
  saved `encoder/config.json` keeps the original checkpoint value (0.0)**, so
  the inference-time configuration and identifiers are unchanged, and
  inference, calibration collection, epoch selection, and temperature fitting
  are all done in eval mode, where dropout is off anyway.
- **`--rdrop-alpha A`** (A >= 0, default 0 = off, requires `--dropout`):
  R-Drop (Liang et al., NeurIPS 2021). Runs two train-mode forward passes per
  micro-batch (each with a different dropout mask), uses the average of the
  selected loss (CE by default; RL + CE for `--method rlcd-grpo`) across both passes, and adds `A x symmetric KL`
  (0.5*(KL(p1||p2)+KL(p2||p1)), computed only over each question's valid
  options, averaged across questions). The grad-accum division, gradient
  clipping, and the per-optimizer-step `torch.mps.empty_cache()` are
  unchanged.
- **Cost (measured 2026-09-24, M4 Pro 24GB, 64 train rows from a real export,
  batch 4, 48 micro-steps)**: no flags, 1.34 seconds/step and about 6.9GB MPS
  memory; `--dropout 0.1` alone, 1.50 seconds/step (about 1.1x) and about
  6.4GB; `--dropout 0.1 --rdrop-alpha 1.0`, 3.15 seconds/step (about **2.35x**)
  and about **11.2GB**. R-Drop does two forward/backward passes, so it costs
  roughly 2x or more in compute, and memory also grows substantially from
  holding both passes' graphs at once. Check memory with `--max-steps` before
  a full-scale run.
- **Recording**: both values are recorded in `local_training_run.json`
  (`dropout`, `rdrop_alpha`) and in `training_metadata.json`'s
  `hyperparameters` (`null` when off), and are also printed on the
  `[local] device=...` startup line. With both flags off, the default path is
  keeps the single-forward path for the selected objective
  (`tests/test_laya_kit_regularization.py`).
- **Caution (MPS)**: running an encoder with attention dropout enabled in
  **train mode plus `torch.no_grad()`** makes PyTorch's MPS SDPA raise
  `NotImplementedError`. The training path (grad enabled) and eval-mode
  path both work fine, so do not add a no-grad evaluation while staying in
  train mode.

## Setting the epoch count and resuming after a stop (`--epochs`, `--resume`, `--keep-resume`, `--local` only, 2026-09-25)

- **Background**: a full-export (18,402 rows, batch 4 x grad-accum 16) local
  MPS training run takes about 1.85 hours per epoch, and we do not leave it
  running overnight (the laptop fan noise). A 6-epoch run spans two days, so
  it needs to be stoppable and resumable epoch by epoch.
- **`--epochs N`** (1 <= N <= 20, default 4 = the previous behavior): sets both
  the number of training loop passes and the full length of the cosine LR
  schedule (`T_max = (number of training items // (batch x grad-accum)) * N`).
  Recorded in `local_training_run.json`'s `epochs` and
  `training_metadata.json`'s `hyperparameters.epochs`. Rejected on the DDP
  path.
- **What gets saved** (always under `--local`, right after every completed
  epoch's `[select]` decision): under `<output-dir>/resume/epoch-000N/`,
  `training_state.pt` (model state_dict, AdamW state, scheduler state,
  Python/NumPy/torch CPU/MPS RNG state), `best_model.pt` (the weights of the
  epoch with the best dev agreement so far -- now kept on disk instead
  of RAM; if the current epoch is not a new best, the previous epoch's file is
  hard-linked forward), and `state.json` (number of completed epochs,
  global_step, per-epoch agreement, the best epoch/score, cumulative training
  time, run configuration). Everything is written to a temporary directory,
  renamed, and `resume/LATEST` is swapped atomically, so if the save is
  interrupted, LATEST always points at a fully intact previous epoch. Only the
  most recent epoch's files are kept; the rest are deleted.
- **Run-configuration consistency check**: if any value in `state.json`'s
  configuration (export manifest/train/dev/calibration file hashes, split and
  provenance fingerprints, exporter/dataset version, base model weights/config
  and tokenizer hashes, Python/torch/Laya runtime identity, method/input-fit,
  training script hash, epochs, batch/grad-accum, device/precision, dropout,
  rdrop-alpha, select-best-epoch and model lengths)
  differs from the current arguments, `--resume` prints the list of
  mismatches and stops. `--max-steps`, `--keep-resume`, and the admission-check
  options are not compared. Changing the kit's code also breaks resumability
  (the script sha256 changes) -- do not modify the kit while a run is paused.
- **Preprocessing cache**: with `--resume`, if
  `<output-dir>/train_items.pt`/`dev_items.pt`/`calib_items.pt` exist and their content
  digest matches what the interrupted run recorded, preprocessing is skipped
  (export consistency was already confirmed by the check above). If they
  differ or are missing, preprocessing runs again, and if the resulting
  digest still does not match the record, the run stops instead of resuming.
- **The resumable unit is the epoch**: if interrupted mid-epoch, the run
  resumes from the last completed epoch and redoes the next epoch from
  scratch (progress within the interrupted epoch is discarded). If
  interrupted before the first epoch finishes, there is no state to resume
  from. On resume, the shuffles of completed epochs are replayed and RNG
  state is restored, so on CPU, the final weights/`epoch_agreements`/
  `selected_epoch` are bit-for-bit identical between an uninterrupted run and
  a resumed one (`tests/test_laya_kit_resume.py`). MPS has no such guarantee
  due to kernel non-determinism. A freshly started `--local` run fixes the
  torch seed at 42 (previously it was not fixed).
- **Safety**: if resume state exists under the same `--output-dir` and
  `--resume` is not passed, a new run refuses to start (to avoid overwriting
  an interrupted run). To start fresh, delete `<output-dir>/resume/` yourself.
  If `--resume` is passed but no state exists, the run refuses to start fresh
  and stops instead.
- **Cleanup**: after the final save succeeds (temperature fitting,
  `model.safetensors`, metadata), `resume/` is deleted. With `--keep-resume`,
  it is kept and its size is printed. `training_metadata.json` keeps the
  original `started_at` plus `resumed_at`/`resumed_from_epoch`, and
  `local_training_run.json` keeps the summed `wall_time_s` across all segments
  plus `resumed_from_epoch`.
- **Disk cost**: for multilingual (about 320M parameters),
  `training_state.pt` is about 3.9GB (1.3GB fp32 weights + 2.6GB AdamW state)
  + `best_model.pt` 1.3GB = about 5.2GB steady-state (measured 4.80GiB). At
  the moment of saving, the previous and new epoch briefly coexist, needing
  up to about 9-10GB. Each epoch takes extra time to write this much.
- **Stopping overnight and resuming in the morning** (example: 6 epochs,
  dropout 0.1):
  ```sh
  PY=/Users/jangjiyong/.local/share/laya/.venv/bin/python
  # Use an array (zsh does not word-split an unquoted "$ARGS" string).
  ARGS=(--export-dir exports/<hash>/laya
        --model-dir /Users/jangjiyong/.local/share/laya/models/multilingual/multilingual
        --output-dir <output-directory> --local --device mps --batch-size 4 --epochs 6
        --method supervised --input-fit lossless --dropout 0.1)
  # Day 1: start. In the evening, after seeing "[resume] saved epoch N/6 checkpoint"
  # (or <output-directory>/resume/LATEST updating), press Ctrl-C in the terminal.
  PYTHONUNBUFFERED=1 $PY training/laya-kit/train_from_export.py "${ARGS[@]}" 2>&1 | tee -a train.log
  # Day 2 morning: append only --resume to the same arguments (changing any other
  # argument is refused).
  PYTHONUNBUFFERED=1 $PY training/laya-kit/train_from_export.py "${ARGS[@]}" --resume 2>&1 | tee -a train.log
  ```
  Pressing Ctrl-C mid-epoch loses that epoch's progress (up to about 1.85
  hours), so it is best to stop right after a `[resume] saved epoch` line
  appears. Without `PYTHONUNBUFFERED=1`, output piped to a file is buffered
  and that line appears late. A process started in the background (`&`)
  ignores SIGINT, so Ctrl-C/`kill -INT` will not work -- in that case, `kill`
  (SIGTERM) still stops it cleanly and resumes from the last completed epoch.
- **MPS measurement (2026-09-25, M4 Pro 24GB, real export with 64 train rows +
  423 calibration rows + 8 test rows, `--batch-size 4 --epochs 2 --dropout
  0.1`)**: an uninterrupted run A and a run B that was interrupted with
  SIGTERM mid-epoch-2 and finished with `--resume` had identical
  `epoch_agreements` (0.3599/0.3777) and `selected_epoch` (2). The fp16
  weights differed by 0.003% (max difference 6.1e-5), but two uninterrupted
  runs (A and C) also differed by 0.0025% at the same max difference of
  6.1e-5, which is within the range of MPS run-to-run non-determinism. The
  resume state size was 4.80GiB per epoch, the preprocessing cache was reused
  on resume, and both a configuration mismatch (`--dropout 0.2`) and starting
  without `--resume` while resume state existed were correctly refused
  without touching the existing state.

## Historical `evaluate_checkpoint` bug fix (2026-09-23)

- **Symptom (measured)**: the post-training evaluation step of the run
  measured above failed with
  `[eval] evaluation step failed or was skipped: 'label'`, so no metrics
  (accuracy, etc.) against `test.jsonl` were recorded at all.
- **Cause**: the official notebook's evaluation cell assumes the gold answer
  has `label`/`score` keys, but pointsman export's `exportDataset()`
  (`src/training/dataset.mjs`) only writes `gold[qid]` as
  `{"probabilities": {...}}` (`training/laya-kit/check_export.py`'s
  `check_gold_probabilities()` enforces this shape). `evaluate_checkpoint()`
  read `g_ans["label"]` directly, raising `KeyError('label')` on choice/noul
  questions.
- **Fix**: `resolve_gold_label(q_type, g_ans, keys=None, n_levels=None)`
  derives the gold label the same way `build_training_item()` determines the
  training target label (`target.index(max(target))`, i.e. the argmax of the
  probabilities). If an explicit `label` key is present (e.g. in a
  hand-written fixture), it takes precedence (backward compatible). All three
  branches (choice/noul/score) were changed to use this function, and the
  score type's gold scalar (used for `score_mae`/`within_1_level`) also falls
  back to this argmax level when `g_ans.get("score", ...)` is absent.
- **A second bug this fix surfaced**: after the fix above, a new error
  appeared during measurement:
  `Cannot cast ufunc 'divide' output from dtype('float64') to dtype('int64')`
  -- the gold probabilities for objective/human labels are one-hot
  (`targetDistribution()` produces integers `1`/`0`), so they deserialize as
  Python `int` after the JSON round trip, and the `np.array(...)` built from
  them is inferred as `int64`, so the subsequent in-place `/=` raises a
  casting error. Fixed by explicitly setting `dtype=np.float64` when building
  the `p_probs`/`g_probs`/`p_probs_score` arrays.
- **Verification**: `tests/test_laya_kit_eval_and_epoch_select.py`'s
  `ResolveGoldLabel` (pure logic) and `EvaluateCheckpointRealExportShape`
  (runs `evaluate_checkpoint()` end-to-end against a synthetic `test.jsonl`
  shaped like a real export row plus a fake `Agent`, including an
  integer-valued-probabilities case) reproduced both bugs RED before
  confirming GREEN. The 2026-09-23 M4 Pro `--local --max-steps` smoke test
  confirmed there was no `[eval]` failure log and that
  `training_metadata.json.metrics` was populated.

## Catastrophic forgetting risk from repeated fine-tuning, and the holdout gate

- Each fine-tune changes the candidate weights. Reusing successive candidates
  as the base for new exports can silently degrade
  quality on purposes/workflows the model previously handled well
  (catastrophic forgetting) -- this cannot be detected just by looking at
  training loss or accuracy on the new dataset.
- Because of this risk, project policy allows checkpoint replacement only
  through **explicit `laya promote`**, and requires a **non-regression** check
  against a **fixed, frozen holdout set** and a shadow comparison against the
  active checkpoint before that. The `benchmark_report.json`/
  `training_metadata.json` this kit produces are only raw material for that
  verification -- they are not, by themselves, a quality guarantee.
- The holdout set must stay fixed rather than being redrawn on every training
  run, so that a new checkpoint can be consistently compared on "did it lose
  the ability to solve problems it used to solve." Redrawing the holdout every
  time would make this comparison meaningless.

## Data rights and evidence requirements

- [ ] Upstream `laya` runtime/model: Apache-2.0 (see NOTICE above). Re-confirm
      commercial use/redistribution terms.
- [ ] This kit does not use the `LocalLLaMA/typed-decisions` benchmark itself
      (the official notebook's data source has been replaced by the export
      folder) -- however, that benchmark's own data rights may still need
      separate review independent of this kit (e.g. if reproducing its
      evaluation methodology).
- [ ] For training data (decision/outcome/label) collected by pointsman: does
      the actual entity that produced that data through execution/annotation
      (the operator themselves, a team member, or a specific runner) consent
      to and have the rights for its use for fine-tuning purposes.
- [ ] Does using outputs (probabilities, labels) from a provider that acted as
      teacher (e.g. the TypeSafe Jev API) to fine-tune another model (Laya)
      violate that provider's terms of use -- if this is a cannot-confirm
      item, those provider comparisons/distillation remain unexecuted until
      permitted; do not defer the boundary until promotion.
- [ ] Has the source/export been checked for personal/sensitive information?
      Pattern checks are best-effort; preserve independent synthetic-source
      provenance and do not read raw private captures by default.

## What this kit does not do

- Training runs only through the explicit trainer invocation; registration,
  prediction collection, qualification, comparison and promotion use their
  separate CLI commands. No action starts automatically.
- The official notebook's 8th cell (optional), which publicly uploads a
  checkpoint to the Hugging Face Hub, is not included in this kit -- pointsman's
  checkpoint promotion path is the local `laya register`, not public Hub
  deployment. Refer to that cell in the official notebook separately if
  needed.
