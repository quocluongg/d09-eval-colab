#!/usr/bin/env bash
# Runner v1.1 (Trieu): sanity -> venv + pinned deps -> selfcheck(v1 data) -> run(v2 data) -> pack -> report.
# Stops at the first failing command and leaves its output verbatim in logs/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Windows Python does not understand MSYS paths, so convert arguments handed to it.
if command -v cygpath >/dev/null 2>&1; then
  native() { cygpath -m "$1"; }
else
  native() { printf '%s' "$1"; }
fi

PKG="$ROOT/runner_v1_1"
LOGS="$PKG/logs"
VENV="$PKG/.venv"
mkdir -p "$(native "$LOGS")"
cd "$PKG"

echo "=== [1/7] sanity: line counts ==="
V2_LINES=$(wc -l < data/features.csv | tr -d ' ')
V1_LINES=$(wc -l < selfcheck_v1/data/features.csv | tr -d ' ')
echo "data/features.csv lines=$V2_LINES expected=38465"
echo "selfcheck_v1/data/features.csv lines=$V1_LINES expected=45318"
if [[ "$V2_LINES" != "38465" || "$V1_LINES" != "45318" ]]; then
  echo "ERROR: line counts differ from Trieu's instructions — stopping, not fixing."
  exit 1
fi

echo
echo "=== [2/7] base interpreter ==="
BASE_PY="${BASE_PYTHON:-}"
if [[ -z "$BASE_PY" ]]; then
  # Prefer 3.12 (Trieu's pin). On Windows `python3` can be a newer 3.14 while
  # `python` is 3.12.10, so pick by version, not by name order alone.
  # The venv is created --without-pip and driven by this interpreter's pip, so a
  # candidate without pip (Colab's /usr/bin/python3.12) cannot be used.
  for cand in python3.12 python python3; do
    command -v "$cand" >/dev/null 2>&1 || continue
    if "$cand" -c 'import sys, pip; sys.exit(0 if sys.version_info >= (3, 12) else 1)' >/dev/null 2>&1; then
      BASE_PY="$cand"
      break
    fi
  done
fi
if [[ -z "$BASE_PY" ]]; then
  echo "ERROR: no Python >= 3.12 with pip on PATH (tried python3.12, python, python3). Set BASE_PYTHON explicitly."
  exit 1
fi
echo "interpreter=$BASE_PY"
echo "$BASE_PY --version: $("$BASE_PY" --version 2>&1)"
"$BASE_PY" -c 'import sys; print(sys.version); sys.exit(0 if sys.version_info >= (3, 12) else 1)' || {
  echo "ERROR: Python >= 3.12 is required; the pinned wheels only exist for cp312+."
  exit 1
}

echo
echo "=== [3/7] venv + pinned dependencies ==="
# Colab's Python ships without ensurepip's bundled wheels, so a plain `venv` dies
# while seeding pip. Create it empty and drive it with the host pip's --python.
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) VPY="$VENV/Scripts/python.exe" ;;
  *)                    VPY="$VENV/bin/python" ;;
esac
# A venv left behind by a failed run still has a usable interpreter, but it may
# belong to a different Python than the one now selected (Colab's pip-less
# python3.12 creates a 3.12 venv that the next run would silently inherit).
BASE_MINOR=$("$BASE_PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
if [[ -x "$VPY" ]]; then
  HAVE_MINOR=$("$VPY" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo none)
else
  HAVE_MINOR=none
fi
if [[ "$HAVE_MINOR" != "$BASE_MINOR" ]]; then
  echo "venv python=$HAVE_MINOR base python=$BASE_MINOR -> creating fresh venv"
  "$BASE_PY" -m venv --without-pip --clear "$(native "$VENV")"
else
  echo "reusing venv python=$HAVE_MINOR"
fi
"$BASE_PY" -m pip --python "$(native "$VPY")" install --upgrade pip 2>&1 | tee "$LOGS/install.log"
"$BASE_PY" -m pip --python "$(native "$VPY")" install -r "$(native "$PKG/requirements.lock.txt")" 2>&1 | tee -a "$LOGS/install.log"
"$BASE_PY" -m pip --python "$(native "$VPY")" freeze > "$LOGS/pip_freeze.txt"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$(native "$PKG/src")"
# Without this, Python block-buffers stdout into tee and Colab shows nothing until `run` exits.
export PYTHONUNBUFFERED=1

echo
echo "=== [4/7] selfcheck on selfcheck_v1/data (v1 reference) ==="
"$VPY" -B -m d09_eval selfcheck --data "$(native "$PKG/selfcheck_v1/data")" 2>&1 | tee "$LOGS/selfcheck.log"

echo
echo "=== [5/7] run: config_tuned on v2 data ==="
RUN_START=$(date +%s)
"$VPY" -B -m d09_eval run \
  --config "$(native "$PKG/config_tuned/experiment.json")" \
  --data "$(native "$PKG/data")" \
  --out "$(native "$PKG/out")" \
  --models logistic_regression random_forest xgboost isolation_forest mlp autoencoder xgboost_if_feature combo_rank_mean combo_or 2>&1 | tee "$LOGS/run.log"
RUN_SECONDS=$(( $(date +%s) - RUN_START ))
printf '%s\n' "$RUN_SECONDS" > "$LOGS/run_seconds.txt"
echo "run wall_seconds=$RUN_SECONDS"

echo
echo "=== [6/7] pack ==="
"$VPY" -B -m d09_eval pack --out "$(native "$PKG/out")" 2>&1 | tee "$LOGS/pack.log"

echo
echo "=== [7/7] report ==="
"$VPY" -B - "$(native "$PKG")" "$(native "$LOGS")" "$(native "$PKG/out")" "$RUN_SECONDS" "$BASE_PY" <<'PY' | tee "$LOGS/RESULTS.txt"
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

pkg, logs, out = (Path(p) for p in sys.argv[1:4])
run_seconds = int(sys.argv[4])
base_py = sys.argv[5]
zip_path = out.parent / f"{out.name}.zip"


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
    ["git", "rev-parse", "--short", "HEAD"], cwd=pkg.parent, capture_output=True, text=True
).stdout.strip() or "not-a-git-checkout"
base_version = subprocess.run(
    [base_py, "--version"], capture_output=True, text=True
).stdout.strip() or subprocess.run([base_py, "--version"], capture_output=True, text=True).stderr.strip()

print("=== SELFCHECK (verbatim) ===")
print((logs / "selfcheck.log").read_text(encoding="utf-8").rstrip("\n"))
print()
print("=== RUN (verbatim) ===")
print((logs / "run.log").read_text(encoding="utf-8").rstrip("\n"))
print()
print("=== PACK (verbatim) ===")
print((logs / "pack.log").read_text(encoding="utf-8").rstrip("\n"))
print()
print("=== DELIVERABLES (What to Send Back) ===")
print(f"1_zip_path={zip_path}")
print(f"1_zip_bytes={zip_path.stat().st_size}")
print(f"2_zip_sha256={sha256(zip_path)}")
print("3_selfcheck_text=see SELFCHECK section above")
print(f"5_run_wall_seconds={run_seconds}")
print(f"5_run_wall_hms={run_seconds // 3600:02d}:{(run_seconds % 3600) // 60:02d}:{run_seconds % 60:02d}")
print()
print("=== MACHINE ===")
print(f"4_os={os_name()}")
print(f"platform={platform.platform()}")
print(f"4_ram_total_gb={ram / 1024**3:.2f}" if ram else "4_ram_total_gb=unknown")
print(f"cpu={cpu_name()}")
print(f"cpu_logical_cores={os.cpu_count()}")
print(f"python={sys.version.split()[0]}")
print(f"base_interpreter={base_py} ({base_version})")
print(f"library_versions={json.dumps(libs, sort_keys=True)}")
print(f"repo_commit={commit}")
PY

echo
echo "Done. Report: $LOGS/RESULTS.txt"
echo "Send back the zip untouched: $PKG/out.zip"