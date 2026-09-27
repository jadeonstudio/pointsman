# training/laya-kit -- Laya Kaggle T4x2 training preparation kit

The `pointsman` repository does not do online learning or automatic
fine-tuning/checkpoint promotion (this is a hard project policy). The official
2xT4 DDP path (`train_from_export.py`, torchrun+NCCL) is **prep material** a
human must upload to Kaggle (or an equivalent 2x NVIDIA T4 DDP machine) and run
by hand. With the `--local` flag (see "Can this run locally on a Mac (MPS)?"
below), the same script **actually runs training** as a single process on
Apple Silicon MPS (or CPU), without torchrun/NCCL/DDP -- even in that case,
checkpoint registration (`laya register`) and promotion are still explicitly
approved by a human.

Pinned official reference:
- Repository commit: `NandhaKishorM/laya@42626c348753fbb17572a813127df2278a1ec527`
- Notebook: `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`
- License: the `LICENSE` at that commit is Apache-2.0. There is no separate
  `NOTICE` file at that same commit (confirmed by browsing the tree).
- `laya`'s own version: `pyproject.toml` at that commit declares `version = "0.3.4"`.

## File layout

| File | Role |
|---|---|
| `check_export.py` | standard-library-only script that inspects an export folder (`train/calibration/test.jsonl` + `manifest.json`) without training |
| `train_from_export.py` | the official notebook's DDP training/eval procedure, adapted to take an export folder as input. A human runs it on Kaggle T4x2 |
| `requirements.lock` | the package list the notebook installs, pinned to the extent verifiable |
| `NOTICE` | upstream Apache-2.0 notice |

## Full flow

```
pointsman dataset export --version <hash> --format laya
        |  (exports/<hash>/laya/{train,calibration,test}.jsonl + manifest.json)
        v
   check_export.py --export-dir exports/<hash>/laya   <- local, no training
        |  (only proceed to the next step if this passes)
        v
   upload the export folder to a Kaggle Notebook (GPU T4 x2, Internet On)
        |
        v
   train_from_export.py --export-dir <uploaded export path> \
       --output-dir /kaggle/working/laya_finetuned_typed_decisions
        |  (torchrun --nproc_per_node=2 DDP training + calibration + evaluation)
        v
   download the checkpoint (model.safetensors, encoder/, tokenizer/,
              rl_agent_config.json, training_metadata.json)
        |
        v
   pointsman laya register   <- register the checkpoint in providers.json (operator)
        v
   pointsman holdout freeze  <- freeze the fixed holdout set (operator)
        v
   pointsman ... qualify     <- generate qualification results
        v
   pointsman ... compare     <- shadow-compare against the current active checkpoint
        v
   pointsman ... promote     <- run only **after explicit owner approval**
```

The exact subcommand arguments for `pointsman laya register`/`holdout
freeze`/`qualify`/`compare`/`promote` follow whatever the CLI looked like when
this kit was built -- this README only guarantees the order; check each
command's flags with `pointsman --help` at the time you run it (do not guess
and write them down here).

## Step-by-step (human-run) -- full runbook for the multilingual checkpoint

Owner decision (2026-09-23): the fine-tuning baseline checkpoint is
**multilingual** (mmBERT, `max_len=1024` / `head_max_len=256`). The HF repo
`convaiinnovations/laya` splits checkpoints into subfolders -- confirmed
locally by checking that every file path under
`~/.local/share/laya/models/multilingual/.cache/huggingface/download/` starts
with `multilingual/` (whereas the same path for the english checkpoint has no
subfolder and sits directly at the root). This kit could not browse the HF
repo's file tree directly, so the subfolder name is given explicitly by a
human via `--model-subdir`, as shown below.

0. **(local) build the distillation dataset**
   ```sh
   pointsman laya distill build --run d3k
   pointsman dataset export --version <hash> --format laya
   ```
   `<hash>` is the `dataset_version` printed by `laya distill build`. The
   exact subcommand arguments follow whatever the CLI looked like when this
   kit was built (see the note above); check the current flags with
   `pointsman training --help` (`TRAINING_HELP`, `src/training/cli.mjs`) at
   the time you run it.

1. **validate the export (local, no GPU needed)**
   ```sh
   python3 training/laya-kit/check_export.py --export-dir exports/<hash>/laya
   ```
   If this fails, do not proceed to the next step.

2. **set up the Kaggle Notebook**
   - Kaggle -> Datasets -> New Dataset: upload the whole `exports/<hash>/laya/`
     folder as a **private** Kaggle Dataset (keep it private; do not make it
     public).
   - Notebook options -> Accelerator: `GPU T4 x2`
   - Notebook options -> Internet: `On` (needed to download the model/packages)
   - Attach all of `training/laya-kit/` and the private Dataset you just
     created to the Kaggle Notebook's Input/Working directory (Add Input).

3. **install dependencies** (in a Kaggle Notebook cell, not locally)
   ```sh
   pip install -q -U laya==0.3.4 transformers datasets safetensors huggingface_hub pyarrow pandas scipy accelerate tabulate
   ```
   For the packages without a pinned version, see the reason marked "cannot
   confirm" in `requirements.lock`.

4. **download the model and run training**
   ```sh
   python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('convaiinnovations/laya'))"
   python3 training/laya-kit/train_from_export.py \
       --export-dir /kaggle/input/<uploaded-export>/laya \
       --model-dir <snapshot path printed above> \
       --model-subdir multilingual \
       --output-dir /kaggle/working/laya_finetuned_multilingual
   ```
   - If `--model-dir` is omitted, the script calls
     `snapshot_download("convaiinnovations/laya")` itself (needs network, so
     Kaggle only). `--model-subdir multilingual` uses the `multilingual/`
     subfolder under that snapshot root as the actual checkpoint directory --
     if that subfolder is missing `rl_agent_config.json`/`tokenizer/`/`encoder/`,
     the script aborts with a clear error before training starts
     (`validate_resolved_model_dir`). For checkpoints like english that sit
     directly at the root with no subfolder, omit `--model-subdir` (existing
     behavior is unchanged).
   - If the export manifest's `exporter_version`/`upstream_contract`/`loader`/
     `source_data_sha256` do not match what this script expects, it aborts
     before training.
   - Before preprocessing each row, `fit_task_head()` -- copied byte-for-byte
     from `workers/laya_worker.py` -- shrinks that row's `state.task` to the
     longest prefix that fits, appending a `" ...[truncated]"` marker (every
     other state key and every question is left untouched), so training sees
     the same input shape that inference actually receives under
     `laya.inputFit:'task-head'`. The `[input-fit] N states task-head
     truncated` log line is the number of states that were shortened this way.
   - Even if the base checkpoint's own `rl_agent_config.json` has
     `max_len`/`head_max_len` different from the final values used at training
     time (1024/256, the override from official notebook cell 4) -- e.g. the
     english checkpoint has 512/192 -- preprocessing (`fit_task_head`/
     `build_training_item`) always judges admission against the final values
     actually used for training (`resolve_effective_cfg`). For the
     multilingual/typed-decisions checkpoint this correction is a no-op, since
     it is already 1024/256.
   - The tokenizer admission (truncation) check runs first, and if the
     fraction of truncated rows exceeds `--max-truncated-fraction` (default
     2%), it aborts. Pass `--allow-truncation` to force it through. (This
     check still catches the remaining truncation cases `fit_task_head`
     cannot fix -- rows where the questions/options alone are too large, or no
     prefix fits at all.)
   - To check only the manifest/tokenizer admission without training, use
     `--dry-run` (does not run torchrun). Locally, even this check cannot pass
     because the `datasets` package is not installed (the local laya `.venv`
     is inference-only -- confirmed with `pip list`); it is only meaningful
     after the Kaggle `pip install` in step 3.
   - Internally, the script runs a `torchrun` invocation of this shape (you do
     not need to type this yourself -- shown here for debugging reference):
     ```sh
     torchrun --standalone --nproc_per_node=2 <output-dir>/train_ddp.py \
         <resolved-model-dir> <output-dir> <output-dir>/train_items.pt <output-dir>/calib_items.pt \
         <derived-model-name> <base-model-dir-name> <exporter_version from the export manifest>
     ```
     `<derived-model-name>` becomes the saved checkpoint's `model_name`: it is
     the base checkpoint's own `rl_agent_config.json.model_name` (or, if
     absent, `--model-subdir` or the model folder name) with `-pointsman-ft`
     appended (`derive_model_name`) -- e.g. if multilingual's base
     `model_name` is `"rl-agent"`, the result is `"rl-agent-pointsman-ft"`. It
     is no longer hard-coded to `"laya-typed-decisions"` as before.
   - **Expected time**: the official notebook's cell 9 markdown states "~4 to
     6 minutes total" for just the `torchrun` training loop (cell 10) alone,
     in minutes. The total GPU-time for the whole notebook (install,
     download, preprocessing, evaluation included) is not documented
     officially -- left as **cannot confirm**. Check your Kaggle account's
     current GPU quota directly before running.

5. **download the outputs**
   Download the following from under `output-dir` to your local machine:
   `model.safetensors`, `encoder/`, `tokenizer/`, `rl_agent_config.json` (now
   also recording `model_name`/`base_model_dir_name`/`exporter_version`),
   `benchmark_report.json` (if evaluation succeeded), and
   `training_metadata.json` (notebook commit, laya version, export hash,
   `base_model_dir_name`, `derived_model_name`, hyperparameters, start/end
   time).

6. **(local) register the checkpoint -> freeze holdout -> qualify -> compare -> promote**
   The exact subcommand flags are copied verbatim from `TRAINING_HELP` in
   `src/training/cli.mjs` (also viewable with `pointsman training --help`). Do
   not invent them.
   ```sh
   pointsman laya register --checkpoint <downloaded absolute path> \
       --python /Users/jangjiyong/.local/share/laya/.venv/bin/python \
       --device mps --precision fp16 --input-fit task-head
   pointsman laya holdout freeze --dataset <hash>
   pointsman laya qualify --candidate <candidate-hash> --dataset <hash> --holdout <holdout-id>
   pointsman laya compare --candidate <candidate-hash> --holdout <holdout-id>
   pointsman laya promote --candidate <candidate-hash> --holdout <holdout-id>
   ```
   **Do not run promote without the owner's explicit approval.** Automatic
   promotion is not implemented in this repository.
   - If `laya register` is not given `--model NAME`, the name defaults to
     `laya/<first 12 chars of the checkpoint hash>`
     (`src/training/laya-lifecycle.mjs` `registerCheckpoint`). The stored
     `rl_agent_config.json.model_name` (the value `derive_model_name` produced)
     is only a record inside the checkpoint -- it is not used as the
     registered name.
   - **Restarting the resident server is not required.** In `src/inference.mjs`,
     `createLayaClient().start()` compares the providers.json laya block's
     hash (`digest(l)`) against the previous child's identity on every infer
     call, and when `promote` changes the checkpoint, it automatically brings
     down the old worker and restarts with the new checkpoint
     (`LAYA_CONFIG_CHANGED`) -- `src/laya-server.mjs`'s `handleInfer`/
     `handlePrepare` also re-read providers.json on every request. This was
     confirmed by reading the code directly. However this automatic pickup
     only holds **while the resident server process is running code that
     knows about `inputFit` (after commit 30aeab4)**. A server started before
     that commit does not send `inputFit` to the worker, so task-head is not
     applied -- restart it once the first time you use a task-head checkpoint.
     After that, restarting is optional:
     ```sh
     launchctl kickstart -k gui/$(id -u)/com.pointsman.laya
     ```

## Expected time (with sources)

- The official notebook's cell 9 markdown states "~4 to 6 minutes total" for
  just the `torchrun` DDP training step (cell 10) -- **in minutes**. This
  kit's training code carries that loop over unchanged, so the training loop
  itself is expected to take a similar order of magnitude.
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

## Can this run locally on a Mac (MPS)? -- `--local` mode (implemented and measured 2026-09-23)

- The official procedure is **CUDA DDP only**, using the `torch.distributed`
  NCCL backend and `torchrun --nproc_per_node=2`. NCCL is only for
  communication between NVIDIA GPUs, so this path does not run as-is on
  Apple Silicon's MPS device.
- `train_from_export.py --local` works around this: it reuses the **exact
  same** loss (RLCD+GRPO), optimizer (AdamW with separate encoder/head LR),
  cosine LR schedule, epoch count (default 4, changeable via `--epochs` only
  under `--local` -- see the section below), per-device batch size, and seed,
  as a single process without `torchrun`/NCCL/DDP (`TRAIN_DDP_SCRIPT` was
  refactored internally into `run_training_loop()`/`finalize_and_save()` so
  the DDP path (`main_ddp`, unchanged) and the local path (`main_local`,
  world_size=1/rank=0) call the same functions). It reads the export's
  train/calibration/test.jsonl using only the standard-library `json` module
  (`load_jsonl_rows`), without the `datasets` package (not installed in the
  local laya venv).
- **Example invocation** (multilingual base checkpoint, as used in this repo):
  ```sh
  /Users/jangjiyong/.local/share/laya/.venv/bin/python training/laya-kit/train_from_export.py \
      --export-dir exports/<hash>/laya \
      --model-dir /Users/jangjiyong/.local/share/laya/models/multilingual/multilingual \
      --output-dir <output-directory> \
      --local --device mps
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
  - `fit_task_head()`/`resolve_effective_cfg()` (input-budget fitting /
    admission criteria) apply exactly the same way as on the DDP path (no
    branching).
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
    steps, projected about 1 hour 45 minutes for 4 epochs. **On a 24GB
    machine, batch-size 4 is recommended.**
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
  loading (~26 seconds), calibration temperature fitting, and test evaluation
  (about 0.6 seconds per case) must be added separately, and MPS throttling
  or thermal effects at the real 9,000-sequence scale were not observed, so
  the actual number may differ). Compared to the official notebook's "~4 to 6
  minutes" (2xT4 DDP, world_size=2), a single local MPS process takes about
  7-11x longer, but costs $0 and consumes no Kaggle GPU quota.
- The pointsman training-data spec's example `providers.json` showing
  `"device": "mps"` refers to the local Laya worker's device setting **at
  inference time**, which is separate from this training kit's `--local`
  training path (though they use the same physical device). Do not confuse
  the two.

## Selecting the best checkpoint per epoch (`--select-best-epoch`, default ON, implemented 2026-09-23)

- **Problem (measured)**: the real training run measured on 2026-09-23 (M4
  Pro, multilingual base, real distilled data with 7,659 train rows, 4
  epochs, batch 4 x grad-accum 16) overfit badly by the last epoch -- train
  argmax agreement (intent/difficulty/risk) was about 0.99/0.92/0.95, but
  held-out test was about 0.72/0.58/0.60. The old behavior of always saving
  the last epoch's weights risks passing a suboptimal checkpoint on to
  promotion.
- **Behavior**: at the end of every epoch (shared between DDP and `--local`,
  via `run_training_loop()`'s `epoch_end_fn` callback), `train_from_export.py`
  puts the model in eval mode and computes per-question-type
  (choice/score/noul) argmax agreement (`compute_calib_agreement()`) against
  `calib_items.pt` (the same real held-out `calibration.jsonl` from the export
  that `finalize_and_save()` uses for temperature fitting), logging
  `[select] epoch N calib_agreement choice=.. score=.. noul=.. mean=..`. If
  the mean (`mean`) across the three types improves on the previous best, that
  epoch's state_dict is copied to and held on CPU (about 1.3GB fp32 for 322M
  parameters, released right after being loaded when training ends). After
  training finishes, the best epoch's state_dict is loaded into the model
  before running `finalize_and_save()` (temperature fitting + saving) -- so
  the saved checkpoint is always "the epoch with the best calibration
  agreement." Under DDP, only rank 0 evaluates, and all ranks synchronize with
  a barrier at the end of every epoch.
- **Flags**: `--select-best-epoch` (default ON) / `--no-select-best-epoch`
  (always keep the last epoch, as before). Both `--local` and DDP are
  supported (passed identically to both paths via `load_common_argv()`'s
  shared argv slots).
- **Metadata**: the training script writes
  `<output-dir>/epoch_selection.json` (`select_best_epoch`, `selected_epoch`,
  and the per-epoch agreement table `epoch_agreements`), and the outer
  `train_from_export.py`'s `main()` reads it and folds it into
  `training_metadata.json`'s `epoch_selection`/`selected_epoch` fields.
- **Verification**: covered by
  `tests/test_laya_kit_eval_and_epoch_select.py`'s `ComputeCalibAgreement`
  (pure aggregation logic) and `TrainDdpScriptEpochSelectionStructure` (a
  static check of the `epoch_end_fn`/best-state logic embedded in
  `TRAIN_DDP_SCRIPT`, plus parity with the outer module). The 2026-09-23 M4
  Pro `--local --max-steps` smoke test confirmed the
  `[select] epoch 1 calib_agreement choice=... score=... noul=... mean=...`
  log line and the `selected_epoch` record in `epoch_selection.json`/
  `training_metadata.json`.

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
  existing loss (RL + CE) across both passes, and adds `A x symmetric KL`
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
  bit-for-bit identical to before in loss and weight updates
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
  epoch with the best calibration agreement so far -- now kept on disk instead
  of RAM; if the current epoch is not a new best, the previous epoch's file is
  hard-linked forward), and `state.json` (number of completed epochs,
  global_step, per-epoch agreement, the best epoch/score, cumulative training
  time, run configuration). Everything is written to a temporary directory,
  renamed, and `resume/LATEST` is swapped atomically, so if the save is
  interrupted, LATEST always points at a fully intact previous epoch. Only the
  most recent epoch's files are kept; the rest are deleted.
- **Run-configuration consistency check**: if any value in `state.json`'s
  configuration (export `manifest.json`/`train.jsonl`/`calibration.jsonl`
  sha256, exporter/dataset version, base model path and `model.safetensors`
  sha256, training script sha256, epochs, batch/grad-accum, device,
  mps-autocast, dropout, rdrop-alpha, select-best-epoch, max_len/head_max_len)
  differs from the current arguments, `--resume` prints the list of
  mismatches and stops. `--max-steps`, `--keep-resume`, and the admission-check
  options are not compared. Changing the kit's code also breaks resumability
  (the script sha256 changes) -- do not modify the kit while a run is paused.
- **Preprocessing cache**: with `--resume`, if
  `<output-dir>/train_items.pt`/`calib_items.pt` exist and their content
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
        --output-dir <output-directory> --local --device mps --batch-size 4 --epochs 6 --dropout 0.1)
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

## `evaluate_checkpoint` post-training evaluation bug fix (2026-09-23)

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

- Every fine-tune run overwrites the previous checkpoint's weights with a new
  training run. Repeatedly fine-tuning on new exports can silently degrade
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

## Data rights/license review checklist (owner must confirm before promote)

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
      item, it must be confirmed before promote.
- [ ] Has a human reviewed the export for personal/sensitive information (the
      project's pattern-based checks are not a complete DLP solution).

## What this kit does not do

- It does not perform the actual model download, pip install, GPU training,
  or checkpoint registration/promotion for you.
- The official notebook's 8th cell (optional), which publicly uploads a
  checkpoint to the Hugging Face Hub, is not included in this kit -- pointsman's
  checkpoint promotion path is the local `laya register`, not public Hub
  deployment. Refer to that cell in the official notebook separately if
  needed.
