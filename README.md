# D09 — model comparison, 1 click trên Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quocluongg/d09-eval-colab/blob/main/colab/run_all.ipynb)

Repo này gói bộ eval `d09_eval` (trong `package/`) kèm dữ liệu (`data/data/`) và một pipeline chạy trọn vẹn từ đầu đến cuối: **selfcheck → run → pack**. Mở notebook bằng badge ở trên rồi `Runtime` → `Run all` (Ctrl+F9) là xong, không phải chỉnh gì.

> Public repo, one-click Colab runner for the D09 five-model comparison. Open the badge above and choose *Runtime → Run all*. The pipeline stops at the first failing command and prints the error verbatim.

## Pipeline làm gì

| Bước | Lệnh | Kết quả |
| --- | --- | --- |
| 1 | `python -m venv .venv` + `pip install -r package/requirements.lock.txt` | Môi trường tách biệt, đúng phiên bản pin |
| 2 | `d09_eval selfcheck` | Train lại baseline LR, so với `reference/lr_v1_reference.json`, dung sai `1e-6` |
| 3 | `d09_eval run` | Train 5 model trên split train, ghi prediction cho split validation |
| 4 | `d09_eval pack` | Nén `out/` thành `out.zip` + `MANIFEST.json` chứa sha256 từng file |
| 5 | report | `logs/RESULTS.txt`: selfcheck nguyên văn, sha256 zip, OS/RAM/CPU, thời gian run |

5 model: `logistic_regression`, `random_forest`, `xgboost`, `isolation_forest`, `mlp`.
4 model có seed `[42, 43, 44, 45, 46]`, LR chạy 1 lần → **21 lượt train**, 21 `predictions.csv` + 21 `run_meta.json`.

Dữ liệu: 45.317 dòng hoá đơn. Split theo ngày (`decidedAt`, fallback `issueDate`): train `< 2026-07-01` (35.896 dòng, 3.773 positive), validation `[2026-07-01, 2026-10-01)` (9.421 dòng, 1.010 positive). Split test bị khoá — runner từ chối trừ khi có `--allow-test` **và** `TEST_UNLOCK.txt`.

## Cấu trúc

```
package/            bộ eval gốc, giữ nguyên byte (xem package/README.md)
  experiment.json   cấu hình 5 model + seed + split
  experiment.sha256 hash của experiment.json, runner tự kiểm
  requirements.lock.txt 17 dependency pin cứng
  reference/lr_v1_reference.json  baseline LR để selfcheck đối chiếu
  src/d09_eval/     runner, selfcheck, pack, score
  tests/            test của bộ eval
data/data/
  features.csv      45.317 dòng
  MANIFEST.json     sha256 của features.csv, runner tự kiểm
colab/
  run_all.ipynb     notebook 1-click
  run_all.sh        toàn bộ pipeline, set -euo pipefail
```

## Chạy local (Windows / macOS / Linux)

Cần Python **3.12.10** (`package/README.md` yêu cầu `>= 3.12`; wheel pinned chỉ có cho cp312+).

```bash
bash colab/run_all.sh
```

Script tự tạo `.venv`, cài lock, chạy cả 3 lệnh và ghi log vào `logs/`. Đổi interpreter nền bằng `BASE_PYTHON=/path/to/python3.12`.

- Windows: `.venv/Scripts/python.exe`
- macOS/Linux: `.venv/bin/python`

Muốn chạy thủ công từng lệnh thì làm theo `package/README.md`.

## Chạy xong gửi lại 4 thứ

1. **`out.zip`** do lệnh `pack` tạo — không giải nén, không đổi tên.
2. **SHA-256 của zip** — dòng `zip_sha256=` trong `logs/RESULTS.txt`. Trên Windows đối chiếu bằng:
   ```powershell
   Get-FileHash out.zip -Algorithm SHA256
   ```
3. **Nguyên văn các dòng `selfcheck`** — mục `=== SELFCHECK (verbatim) ===` trong `logs/RESULTS.txt` (hoặc `logs/selfcheck.log`).
4. **Máy + thời gian** — `os=`, `ram_total_gb=`, `cpu=`, `run_wall_seconds=` trong `logs/RESULTS.txt`.

Lệnh nào báo lỗi thì **dừng** và gửi nguyên văn lỗi, không tự sửa. `set -euo pipefail` trong `run_all.sh` và `check=True` trong notebook bảo đảm chuyện đó: không có bước nào chạy tiếp sau khi bước trước thất bại.

## Không được sửa

`package/experiment.json`, `package/experiment.sha256`, `package/requirements.lock.txt`, `package/reference/`, `package/src/`, `package/tests/`, và mọi file trong `data/data/`.

Runner tự kiểm sha256 của config và của `features.csv` trước khi chạy; lệch một byte là dừng. `.gitattributes` đặt `* -text` để git không đổi EOL, giữ file nguyên vẹn khi clone về.

## Ghi chú tái lập

- `selfcheck` nhạy với phiên bản thư viện: phải cài đúng `requirements.lock.txt` (numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1, scipy 1.18.1, xgboost 3.4.1). Cài lệch là `max_abs_diff` vượt `1e-6`.
- Colab cần Python 3.12 và Linux x86_64 glibc ≥ 2.28 — tất cả wheel pinned đều có sẵn trên PyPI cho cấu hình đó, không phải build từ source.
- `n_jobs=1` trong config nên runtime tỉ lệ thuận số nhân CPU; Colab 2 vCPU thường nhanh hơn máy local 2 core/4 thread.
- Lệnh `score` (chấm điểm với truth file) không nằm trong pipeline 1-click vì truth file không được phát kèm.