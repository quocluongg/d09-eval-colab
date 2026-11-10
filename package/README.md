# Model Comparison Runner (`d09_eval`)

Small, reproducible evaluation runner so a teammate can run the five-model comparison with a few commands and send back the packaged predictions.

## Installation (3 steps)

1. Install **Python 3.12.10** and open a terminal in `ai/eval/`.
2. Create a local virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Install the pinned dependencies from `requirements.lock.txt`:
   ```bash
   .venv/Scripts/python -m pip install -r requirements.lock.txt
   ```
   *(On macOS/Linux use `.venv/bin/python` instead of `.venv/Scripts/python`.)*

## Commands to Run

Run these three commands in order from the repository root (or inside `ai/eval` with `PYTHONPATH=src`), replacing `<data_dir>` with the path to the unpacked data folder and `<out_dir>` with the output directory:

1. **Verify the environment and baseline reproduction (`selfcheck`):**
   ```bash
   PYTHONPATH=ai/eval/src ai/eval/.venv/Scripts/python -m d09_eval selfcheck --data <data_dir>
   ```
2. **Train all models on the training split and write validation predictions (`run`):**
   ```bash
   PYTHONPATH=ai/eval/src ai/eval/.venv/Scripts/python -m d09_eval run --data <data_dir> --config ai/eval/experiment.json --out <out_dir>
   ```
3. **Package the output directory with a SHA-256 manifest (`pack`):**
   ```bash
   PYTHONPATH=ai/eval/src ai/eval/.venv/Scripts/python -m d09_eval pack --out <out_dir>
   ```

## What to Send Back

- Send back the generated `.zip` archive created by the `pack` command (for example, `out.zip`).

## What NOT to Touch

- Do **not** edit `experiment.json`, `experiment.sha256`, `requirements.lock.txt`, `reference/lr_v1_reference.json`, or any file inside `<data_dir>` (including `features.csv` and `MANIFEST.json`).
- Do **not** edit any source files under `src/` or `tests/`.

If anything fails, stop and send the full error text.
