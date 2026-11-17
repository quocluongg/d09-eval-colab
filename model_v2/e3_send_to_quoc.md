# Instructions for Second Runner (Quoc) - Runner v1.1

## Packages and SHA-256
- `runner_v1_1.zip`: `5a8aa4af663a7bb8154d7bdc791a871580e4520141fb07a0f6fb81e76d179e10`
- `v2_medium_s42_data.zip`: `ebfcc3df6623ea5c642ef43ab7ffff3089bd7561351e04811f11f097d817329f`
- `v1_selfcheck_data.zip`: `aebc11c07c16b4c81c106c7e32a0dc7bfb0c5aebe8342d44b88837d94ac2659f`

## Environment Requirement
Please use **Python 3.12.10** and install dependencies from `requirements.lock.txt`.
If Python 3.12.10 is not available, tell us your version; 3.13 gave the same numbers in the first package, but send the selfcheck text either way.

> **Important:** `selfcheck` must be run on `selfcheck_v1/data` (the v1 data), **not** on `data` (the v2 data), because the reference numbers in `reference/lr_v1_reference.json` belong to v1.
> Both `v2_medium_s42_data.zip` and `v1_selfcheck_data.zip` extract a `data/` folder. Always unpack `v1_selfcheck_data.zip` into its own `selfcheck_v1` subdirectory, never into the same folder as `v2_medium_s42_data.zip`, so the second unpack does not silently overwrite the first.

## Commands to Run (Linux / bash)

```bash
mkdir -p runner_v1_1
unzip -o runner_v1_1.zip -d runner_v1_1
unzip -o v2_medium_s42_data.zip -d runner_v1_1
unzip -o v1_selfcheck_data.zip -d runner_v1_1/selfcheck_v1
cd runner_v1_1

# Sanity check line counts (stop and report if either differs):
wc -l data/features.csv                 # expected: 38465
wc -l selfcheck_v1/data/features.csv    # expected: 45318

python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.lock.txt

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src
.venv/bin/python -B -m d09_eval selfcheck --data selfcheck_v1/data
.venv/bin/python -B -m d09_eval run --config config_tuned/experiment.json --data data --out out --models logistic_regression random_forest xgboost isolation_forest mlp autoencoder xgboost_if_feature combo_rank_mean combo_or
.venv/bin/python -B -m d09_eval pack --out out
```

## Commands to Run (Windows / PowerShell)

```powershell
Expand-Archive -Path runner_v1_1.zip -DestinationPath .\runner_v1_1 -Force
Expand-Archive -Path v2_medium_s42_data.zip -DestinationPath .\runner_v1_1 -Force
Expand-Archive -Path v1_selfcheck_data.zip -DestinationPath .\runner_v1_1\selfcheck_v1 -Force
Set-Location .\runner_v1_1

# Sanity check line counts (stop and report if either differs):
(Get-Content .\data\features.csv | Measure-Object -Line).Lines                 # expected: 38465
(Get-Content .\selfcheck_v1\data\features.csv | Measure-Object -Line).Lines    # expected: 45318

py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt

$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -B -m d09_eval selfcheck --data selfcheck_v1\data
.\.venv\Scripts\python.exe -B -m d09_eval run --config config_tuned\experiment.json --data data --out out --models logistic_regression random_forest xgboost isolation_forest mlp autoencoder xgboost_if_feature combo_rank_mean combo_or
.\.venv\Scripts\python.exe -B -m d09_eval pack --out out
```

## What `selfcheck` Prints on Success
On success, `selfcheck` prints `selfcheck max_abs_diff=... threshold=...`, `selfcheck actual_cm=... reference_cm=...`, and `SELFCheck OK` (exit code 0). The code uses a score-difference tolerance of `1e-6` (`max_abs_diff <= 1e-6`).

## What to Send Back
1. The packed output zip produced by `pack`.
2. Its SHA-256 hash.
3. The full `selfcheck` text output.
4. Your OS and RAM.
5. Total run time.

Please do not open or look for any other file.
