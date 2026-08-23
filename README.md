# QAT Research v2: Expanded Selective-Tensor Quantization Study

Rebuilt from scratch after a laptop reset wiped the original v1 code (the v1
paper text, results, and pipeline design survive only in
`../research paper context.txt` and the two v1 PDFs in `../`). This is the
expansion driven by ACL Submission #173 reviewer feedback -- see
[`docs/PLAN.md`](docs/PLAN.md) for the full reviewer-to-experiment mapping and
[`docs/MATH.md`](docs/MATH.md) for the formal memory-scaling and STE-asymmetry
derivations that back the paper's Discussion section.

## Setup

```bash
cd qat-research
python -m venv .venv
./.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu128
./.venv/Scripts/pip install -r requirements.txt
```

Verify CUDA:
```bash
./.venv/Scripts/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Pipeline (run from `src/`)

1. **Tokenize data** (once per model, tokenizer differs per model family):
   ```bash
   python prepare_data.py --model opt-350m
   python prepare_data.py --model pythia-410m
   python prepare_data.py --model qwen2.5-0.5b
   python prepare_data.py --model frontier
   ```

2. **PTQ baseline family** (FP16 zero-shot, INT8 dynamic, SmoothQuant, AWQ):
   ```bash
   python eval_baseline.py --model opt-350m
   ```

3. **Full selective-QAT matrix** (4 trained configs x 3 seeds, per model):
   ```bash
   python run_matrix.py --model opt-350m
   python run_matrix.py --model pythia-410m
   python run_matrix.py --model qwen2.5-0.5b
   ```
   Resumable: re-running skips seed/strategy combos already in `results/<model>.json`.

4. **Memory-frontier study** (QAT-infeasibility probe + QLoRA comparison):
   ```bash
   python qlora_frontier.py --mode both
   ```

5. **Downstream zero-shot eval** (on trained checkpoints from step 3):
   ```bash
   python eval_downstream.py --model opt-350m --strategy weights_only --seed 42
   ```

6. **Aggregate stats** (mean +/- std, significance test):
   ```bash
   python stats.py --model all
   ```

## Model registry

See [`src/config.py`](src/config.py) for the model roster, hyperparameters,
and seeds -- single source of truth, referenced by every script.

## Results layout

- `results/<model_key>.json` -- all runs for a model, keyed by config name,
  each a list (one entry per seed).
- `results/<model_key>_downstream_<strategy>.json` -- lm-eval-harness output.
- `checkpoints/<model>_<strategy>_seed<seed>.pt` -- FP16 state_dicts, loaded
  by `eval_downstream.py` for downstream eval (never re-evaluate a fresh base
  model and call it a QAT result -- always load the actual trained weights).

## Reproducibility

Every result entry in `results/*.json` carries its own seed, step count,
batch size, gradient-accumulation steps, and learning rate inline -- no
separate hyperparameter log to go stale. `docs/PLAN.md` documents the
experiment design decisions and the reviewer feedback each addresses.
