#!/usr/bin/env bash
# One-click end-to-end: venv -> pinned deps -> selfcheck -> run -> pack -> report.
# Stops at the first failing command and leaves its output verbatim in logs/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Windows Python does not understand MSYS paths, so convert arguments handed to it.
if command -v cygpath >/dev/null 2>&1; then
  native() { cygpath -m "$1"; }
else
  native() { printf '%s' "$1"; }
fi

PKG="$ROOT/package"
DATA="$ROOT/data/data"
OUT="$ROOT/out"
LOGS="$ROOT/logs"
VENV="$ROOT/.venv"
mkdir -p "$LOGS"

BASE_PY="${BASE_PYTHON:-python}"

echo "=== [1/6] base interpreter ==="
"$BASE_PY" -c 'import sys; print(sys.version); sys.exit(0 if sys.version_info >= (3, 12) else 1)' || {
  echo "ERROR: Python >= 3.12 is required (package/README.md pins 3.12.10)."
  exit 1
}

echo
echo "=== [2/6] venv + pinned dependencies ==="
if [[ ! -d "$VENV" ]]; then
  "$BASE_PY" -m venv "$(native "$VENV")"
fi
if [[ -x "$VENV/Scripts/python.exe" ]]; then
  VPY="$VENV/Scripts/python.exe"
else
  VPY="$VENV/bin/python"
fi
"$VPY" -m pip install -r "$(native "$PKG/requirements.lock.txt")" 2>&1 | tee "$LOGS/install.log"
"$VPY" -m pip freeze > "$LOGS/pip_freeze.txt"

export PYTHONPATH="$(native "$PKG/src")"
# Without this, Python block-buffers stdout into tee and Colab shows nothing until `run` exits.
export PYTHONUNBUFFERED=1

echo
echo "=== [3/6] selfcheck ==="
"$VPY" -m d09_eval selfcheck \
  --data "$(native "$DATA")" \
  --config "$(native "$PKG/experiment.json")" \
  --reference "$(native "$PKG/reference/lr_v1_reference.json")" 2>&1 | tee "$LOGS/selfcheck.log"

echo
echo "=== [4/6] run: 5 models on train split, validation predictions ==="
RUN_START=$(date +%s)
"$VPY" -m d09_eval run \
  --data "$(native "$DATA")" \
  --config "$(native "$PKG/experiment.json")" \
  --out "$(native "$OUT")" 2>&1 | tee "$LOGS/run.log"
RUN_SECONDS=$(( $(date +%s) - RUN_START ))
printf '%s\n' "$RUN_SECONDS" > "$LOGS/run_seconds.txt"
echo "run wall_seconds=$RUN_SECONDS"

echo
echo "=== [5/6] pack ==="
"$VPY" -m d09_eval pack --out "$(native "$OUT")" 2>&1 | tee "$LOGS/pack.log"

echo
echo "=== [6/6] report ==="
"$VPY" - "$(native "$ROOT")" "$(native "$LOGS")" "$(native "$OUT")" "$RUN_SECONDS" <<'PY' | tee "$LOGS/RESULTS.txt"
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

root, logs, out = (Path(p) for p in sys.argv[1:4])
run_seconds = int(sys.argv[4])
zip_path = root / f"{out.name}.zip"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def total_ram_bytes():
    if sys.platform.startswith("linux"):
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    if sys.platform == "win32":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return int(st.ullTotalPhys)
    if sys.platform == "darwin":
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
    return None


def os_name():
    if sys.platform == "win32":
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion") as k:
            product = winreg.QueryValueEx(k, "ProductName")[0]
            build = winreg.QueryValueEx(k, "CurrentBuildNumber")[0]
        return f"{product} (build {build})"
    os_release = Path("/etc/os-release")
    if os_release.exists():
        for line in os_release.read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip('"')
    return platform.platform()


def cpu_name():
    if sys.platform == "win32":
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


ram = total_ram_bytes()
meta_files = sorted(out.rglob("run_meta.json"))
libs = json.loads(meta_files[0].read_text(encoding="utf-8"))["library_versions"] if meta_files else {}
commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True
).stdout.strip() or "not-a-git-checkout"

print("=== SELFCHECK (verbatim) ===")
print((logs / "selfcheck.log").read_text(encoding="utf-8").rstrip("\n"))
print()
print("=== RUN (verbatim) ===")
print((logs / "run.log").read_text(encoding="utf-8").rstrip("\n"))
print()
print("=== PACK (verbatim) ===")
print((logs / "pack.log").read_text(encoding="utf-8").rstrip("\n"))
print()
print("=== DELIVERABLES ===")
print(f"zip_path={zip_path}")
print(f"zip_bytes={zip_path.stat().st_size}")
print(f"zip_sha256={sha256(zip_path)}")
print(f"run_wall_seconds={run_seconds}")
print(f"run_wall_hms={run_seconds // 3600:02d}:{(run_seconds % 3600) // 60:02d}:{run_seconds % 60:02d}")
print()
print("=== MACHINE ===")
print(f"os={os_name()}")
print(f"platform={platform.platform()}")
print(f"ram_total_gb={ram / 1024**3:.2f}" if ram else "ram_total_gb=unknown")
print(f"cpu={cpu_name()}")
print(f"cpu_logical_cores={os.cpu_count()}")
print(f"python={sys.version.split()[0]}")
print(f"library_versions={json.dumps(libs, sort_keys=True)}")
print(f"repo_commit={commit}")
PY

echo
echo "Done. Report: $LOGS/RESULTS.txt"
echo "Send back the zip untouched: $ROOT/out.zip"