"""Packaging utility for d09_eval prediction outputs."""

from __future__ import annotations

from pathlib import Path
import zipfile

from d09_eval.runner import _assert_not_forbidden_path, compute_sha256
import json


def pack_outputs(out_dir: Path, zip_path: Path | None = None) -> Path:
    _assert_not_forbidden_path(out_dir)
    if not out_dir.exists() or not out_dir.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {out_dir}")

    files_hash: dict[str, str] = {}
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.json" and not p.name.endswith(".zip"):
            _assert_not_forbidden_path(p)
            rel = p.relative_to(out_dir).as_posix()
            files_hash[rel] = compute_sha256(p)

    manifest_obj = {
        "file_count": len(files_hash),
        "files": files_hash,
    }
    manifest_path = out_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest_obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    target_zip = zip_path if zip_path is not None else out_dir.parent / f"{out_dir.name}.zip"
    _assert_not_forbidden_path(target_zip)
    with zipfile.ZipFile(target_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out_dir.rglob("*")):
            if p.is_file() and not p.name.endswith(".zip"):
                rel = p.relative_to(out_dir).as_posix()
                zf.write(p, arcname=rel)

    zip_sha = compute_sha256(target_zip)
    print(f"pack zip={target_zip} sha256={zip_sha} files={len(files_hash) + 1}")
    return target_zip
