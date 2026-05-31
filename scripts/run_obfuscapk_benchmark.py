#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_OBFUSCATION_SETS: dict[str, list[str]] = {
    "rebuild": ["Rebuild", "NewAlignment", "NewSignature"],
    "rename": ["ClassRename", "MethodRename", "Rebuild", "NewAlignment", "NewSignature"],
    "string_encrypt": ["ConstStringEncryption", "Rebuild", "NewAlignment", "NewSignature"],
    "reflection": ["Reflection", "Rebuild", "NewAlignment", "NewSignature"],
    "call_indirection": ["CallIndirection", "Rebuild", "NewAlignment", "NewSignature"],
    "control_flow": ["Goto", "Reorder", "Rebuild", "NewAlignment", "NewSignature"],
    "junk_code": ["Nop", "ArithmeticBranch", "Rebuild", "NewAlignment", "NewSignature"],
    "manifest_noise": ["RandomManifest", "Rebuild", "NewAlignment", "NewSignature"],
}

ID_CANDIDATES = ("id", "sha256", "ID", "Id")


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]], str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    id_col = next((col for col in ID_CANDIDATES if col in fieldnames), "")
    if not id_col:
        raise ValueError(f"{path} must contain one of id columns: {', '.join(ID_CANDIDATES)}")
    if "label" not in fieldnames:
        raise ValueError(f"{path} must contain label")
    return fieldnames, rows, id_col


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = [
            "sid_original",
            "sha256_original",
            "sha256_obfuscated",
            "label",
            "split",
            "technique",
            "apk_original_path",
            "apk_obfuscated_path",
            "status",
            "reason",
        ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_label_csvs(out_root: Path, index_rows: list[dict[str, Any]]) -> dict[str, str]:
    labels_dir = out_root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    by_technique: dict[str, list[dict[str, str]]] = {}
    for row in index_rows:
        if row.get("status") != "ok" or not row.get("sha256_obfuscated"):
            continue
        technique = str(row.get("technique") or "unknown")
        by_technique.setdefault(technique, []).append(
            {
                "id": str(row["sha256_obfuscated"]).lower(),
                "sha256": str(row["sha256_obfuscated"]).lower(),
                "label": str(row.get("label", "")),
                "sid_original": str(row.get("sid_original", "")),
                "sha256_original": str(row.get("sha256_original", "")),
                "technique": technique,
            }
        )
    paths = {}
    for technique, rows in sorted(by_technique.items()):
        path = labels_dir / f"{technique}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["id", "sha256", "label", "sid_original", "sha256_original", "technique"],
            )
            writer.writeheader()
            writer.writerows(rows)
        paths[technique] = str(path)
    return paths


def _load_obfuscation_sets(path: str) -> dict[str, list[str]]:
    if not path:
        return dict(DEFAULT_OBFUSCATION_SETS)
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    sets = raw.get("obfuscation_sets", raw)
    out: dict[str, list[str]] = {}
    for name, values in sets.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Obfuscation set {name!r} must be a non-empty list")
        out[str(name)] = [str(v) for v in values]
    return out


def _collect_apks(apk_dir: Path) -> list[Path]:
    if not apk_dir.exists():
        raise FileNotFoundError(f"APK directory does not exist: {apk_dir}")
    return sorted(p for p in apk_dir.rglob("*.apk") if p.is_file())


def _match_apks(apk_paths: list[Path], sids: set[str], mode: str) -> dict[str, Path]:
    mode = mode.lower()
    if mode not in {"auto", "stem", "sha256"}:
        raise ValueError("--match-by must be auto, stem, or sha256")

    matched: dict[str, Path] = {}
    if mode in {"auto", "stem"}:
        for path in apk_paths:
            sid = path.stem.lower()
            if sid in sids and sid not in matched:
                matched[sid] = path

    if mode in {"auto", "sha256"}:
        missing = sids - set(matched)
        if missing or mode == "sha256":
            for path in apk_paths:
                sha = _sha256_file(path)
                if sha in sids and sha not in matched:
                    matched[sha] = path
    return matched


def _run_obfuscapk(
    *,
    obfuscapk_bin: str,
    input_apk: Path,
    output_apk: Path,
    work_dir: Path,
    obfuscators: list[str],
    ignore_libs: bool,
    timeout: int,
) -> tuple[bool, str]:
    output_apk.parent.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = shlex.split(obfuscapk_bin)
    if not cmd:
        return False, "empty Obfuscapk command"
    for obfuscator in obfuscators:
        cmd.extend(["-o", obfuscator])
    cmd.extend(["-w", str(work_dir), "-d", str(output_apk)])
    if ignore_libs:
        cmd.append("-i")
    cmd.append(str(input_apk))
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        reason = (proc.stderr or proc.stdout or "").strip().replace("\r", " ").replace("\n", " ")
        return False, reason[:1000] or f"obfuscapk exited with {proc.returncode}"
    if not output_apk.exists() or output_apk.stat().st_size <= 0:
        return False, "obfuscapk finished but output APK is missing or empty"
    return True, ""


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(int(args.seed))
    csv_path = Path(args.csv)
    apk_dir = Path(args.apk_dir)
    out_root = Path(args.out_root)
    split = str(args.split)
    _fieldnames, rows, id_col = _read_csv(csv_path)

    selected = rows
    if args.limit and args.limit > 0 and len(rows) > args.limit:
        selected = random.sample(rows, int(args.limit))
    sids = {str(row.get(id_col, "")).strip().lower() for row in selected}
    sids.discard("")

    apk_paths = _collect_apks(apk_dir)
    sid_to_apk = _match_apks(apk_paths, sids, args.match_by)
    obfuscation_sets = _load_obfuscation_sets(args.obfuscation_config)
    requested = args.techniques or list(obfuscation_sets)
    unknown = [name for name in requested if name not in obfuscation_sets]
    if unknown:
        raise ValueError(f"Unknown obfuscation techniques: {unknown}; available={sorted(obfuscation_sets)}")

    index_rows: list[dict[str, Any]] = []
    for row in selected:
        sid = str(row.get(id_col, "")).strip().lower()
        label = str(row.get("label", ""))
        input_apk = sid_to_apk.get(sid)
        if input_apk is None:
            for technique in requested:
                index_rows.append(
                    {
                        "sid_original": sid,
                        "sha256_original": sid,
                        "sha256_obfuscated": "",
                        "label": label,
                        "split": split,
                        "technique": technique,
                        "apk_original_path": "",
                        "apk_obfuscated_path": "",
                        "status": "failed",
                        "reason": "source APK not found",
                    }
                )
            continue

        for technique in requested:
            obfuscators = obfuscation_sets[technique]
            output_apk = out_root / "apks" / technique / f"{sid}.apk"
            work_dir = out_root / "work" / technique / sid
            if args.resume and output_apk.exists() and output_apk.stat().st_size > 0:
                ok = True
                reason = "resumed_existing_apk"
            else:
                ok, reason = _run_obfuscapk(
                    obfuscapk_bin=args.obfuscapk_bin,
                    input_apk=input_apk,
                    output_apk=output_apk,
                    work_dir=work_dir,
                    obfuscators=obfuscators,
                    ignore_libs=bool(args.ignore_libs),
                    timeout=int(args.timeout),
                )
            sha_obf = _sha256_file(output_apk) if ok and output_apk.exists() else ""
            index_rows.append(
                {
                    "sid_original": sid,
                    "sha256_original": sid,
                    "sha256_obfuscated": sha_obf,
                    "label": label,
                    "split": split,
                    "technique": technique,
                    "obfuscators": " ".join(obfuscators),
                    "apk_original_path": str(input_apk),
                    "apk_obfuscated_path": str(output_apk) if output_apk.exists() else "",
                    "status": "ok" if ok else "failed",
                    "reason": reason,
                }
            )
            if not args.keep_work and work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)

    index_path = out_root / "obfuscated_index.csv"
    _write_csv(index_path, index_rows)
    label_csvs = _write_label_csvs(out_root, index_rows)
    failed = [row for row in index_rows if row["status"] != "ok"]
    summary = {
        "csv": str(csv_path),
        "apk_dir": str(apk_dir),
        "out_root": str(out_root),
        "selected": len(selected),
        "matched_sources": len(sid_to_apk),
        "techniques": requested,
        "ok": len(index_rows) - len(failed),
        "failed": len(failed),
        "index": str(index_path),
        "label_csvs": label_csvs,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if failed and args.fail_on_error:
        raise RuntimeError(f"{len(failed)} Obfuscapk jobs failed; see {index_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a real Obfuscapk robustness benchmark from clean APKs.")
    parser.add_argument("--csv", default="results/labels/test.csv", help="Label CSV for the clean split to sample.")
    parser.add_argument("--apk-dir", default="D:/resource/test", help="Directory containing clean APK files.")
    parser.add_argument("--out-root", default="D:/obfuscapk_benchmark", help="Output benchmark directory.")
    parser.add_argument("--split", default="test_obfuscapk")
    parser.add_argument("--limit", type=int, default=0, help="Optional sample limit; 0 means all matched rows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--match-by", default="auto", choices=["auto", "stem", "sha256"])
    parser.add_argument("--techniques", nargs="+", default=None, help="Subset of obfuscation sets to run.")
    parser.add_argument("--obfuscation-config", default="", help="YAML file with obfuscation_sets mapping.")
    parser.add_argument("--obfuscapk-bin", default="obfuscapk")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--ignore-libs", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip jobs whose output APK already exists.")
    parser.add_argument("--keep-work", action="store_true", help="Keep Obfuscapk work directories for debugging.")
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
