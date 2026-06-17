#! /usr/bin/python

#
# Description:
# ================================================================
# Compare saved fit artifacts across legacy, 2025, and 6gev modes.
# ================================================================
#

import argparse
import ast
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from utility import project_display_path, project_path, safe_tabulate as tabulate


DEFAULT_MODES = ("legacy", "2025", "6gev")
REPORT_DIR_NAME = "fit_mode_comparison"
BW_PARAMETERS = ("k", "gamma", "mass")
TRANSITION_PARAMETERS = ("w_dis_transition", "damping_dis_width")
TABLE_FORMAT = "github"
CANONICAL_STAGE_FILES = (
    "fit_results.csv",
    "full_results.csv",
    "full_results_errors.csv",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare saved a1n_analysis fit artifacts across dataset modes."
    )
    parser.add_argument(
        "--fit-data-root",
        default=project_path("fit_data"),
        help="Fit-data root directory. Defaults to the project fit_data directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where comparison reports will be written. Defaults to fit_data/fit_mode_comparison.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(DEFAULT_MODES),
        help="Modes to compare. Defaults to legacy 2025 6gev.",
    )
    parser.add_argument(
        "--mode-dir",
        action="append",
        default=[],
        help="Override a mode artifact directory with MODE=DIR. DIR may be absolute or relative to fit_data.",
    )
    return parser.parse_args()


def parse_mode_overrides(raw_overrides):
    overrides = {}
    for raw_value in raw_overrides:
        if "=" not in raw_value:
            raise ValueError(f"Invalid --mode-dir value '{raw_value}'. Expected MODE=DIR.")
        mode, directory = raw_value.split("=", 1)
        mode = mode.strip().lower()
        directory = directory.strip()
        if not mode or not directory:
            raise ValueError(f"Invalid --mode-dir value '{raw_value}'. Expected MODE=DIR.")
        overrides[mode] = directory
    return overrides


def resolve_override_path(fit_data_root, raw_path):
    path = Path(raw_path)
    if not path.is_absolute():
        path = fit_data_root / raw_path
    return path.resolve()


def find_splash_files(fit_data_root, mode):
    patterns = {
        "legacy": ["load_data_splash_legacy_*.txt"],
        "2025": ["load_data_splash_*2025*.txt"],
        "6gev": ["load_data_splash_6gev_*.txt", "load_data_splash_12gev_*.txt"],
    }
    splash_files = []
    for pattern in patterns.get(mode, [f"load_data_splash_{mode}_*.txt"]):
        splash_files.extend(sorted(fit_data_root.glob(pattern)))
    unique_files = []
    seen = set()
    for path in splash_files:
        if path not in seen:
            unique_files.append(path)
            seen.add(path)
    return unique_files


def parse_splash_list(raw_value):
    raw_value = str(raw_value).strip()
    if not raw_value or raw_value.lower() == "none":
        return []
    return [item.strip() for item in raw_value.split(", ") if item.strip()]


def format_display_path_list(paths):
    if not paths:
        return "none"
    return ", ".join(project_display_path(path) for path in paths)


def parse_splash_label_counts(raw_value):
    raw_value = str(raw_value).strip()
    if not raw_value or raw_value.lower() == "none":
        return {}

    counts = {}
    for item in raw_value.split(", "):
        if ": " in item:
            label, count = item.rsplit(": ", 1)
            try:
                counts[label] = int(count)
            except ValueError:
                counts[label] = count
        elif item:
            counts[item] = None
    return counts


def parse_splash_file(path):
    parsed = {
        "path": path,
        "dataset_mode": None,
        "analysis_scope": None,
        "frames": {},
    }
    current_frame = None

    with open(path, "r", encoding="utf-8") as handle:
        lines = [line.rstrip("\n") for line in handle]

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("="):
            continue

        if stripped.startswith("[load_data] dataset_mode="):
            match = re.search(r"dataset_mode=([^\s]+)\s+analysis_scope=([^\s]+)", stripped)
            if match:
                parsed["dataset_mode"] = match.group(1)
                parsed["analysis_scope"] = match.group(2)
            continue

        if stripped.startswith("[load_data] ") and stripped.endswith("_df"):
            current_frame = stripped.replace("[load_data] ", "", 1)
            parsed["frames"][current_frame] = {
                "rows": None,
                "sources": [],
                "sources_raw": "none",
                "cuts": [],
                "cuts_raw": "none",
                "labels": {},
                "labels_raw": "none",
            }
            continue

        if current_frame is None:
            continue

        if stripped.startswith("rows:"):
            raw_rows = stripped.split(":", 1)[1].strip()
            try:
                parsed["frames"][current_frame]["rows"] = int(raw_rows)
            except ValueError:
                parsed["frames"][current_frame]["rows"] = raw_rows
        elif stripped.startswith("sources:"):
            raw_value = stripped.split(":", 1)[1].strip()
            parsed["frames"][current_frame]["sources"] = parse_splash_list(raw_value)
            parsed["frames"][current_frame]["sources_raw"] = format_display_path_list(
                parsed["frames"][current_frame]["sources"]
            )
        elif stripped.startswith("cuts:"):
            raw_value = stripped.split(":", 1)[1].strip()
            parsed["frames"][current_frame]["cuts_raw"] = raw_value or "none"
            parsed["frames"][current_frame]["cuts"] = parse_splash_list(raw_value)
        elif stripped.startswith("labels:"):
            raw_value = stripped.split(":", 1)[1].strip()
            parsed["frames"][current_frame]["labels_raw"] = raw_value or "none"
            parsed["frames"][current_frame]["labels"] = parse_splash_label_counts(raw_value)

    return parsed


def load_splash_summaries(splash_files):
    return [parse_splash_file(path) for path in splash_files]


def discover_mode_directory(mode, fit_data_root, overrides):
    fit_data_root = fit_data_root.resolve()
    splash_files = find_splash_files(fit_data_root, mode)
    candidates = []
    note = ""

    if mode in overrides:
        override_path = resolve_override_path(fit_data_root, overrides[mode])
        candidates = [override_path]
        note = f"user override -> {override_path}"
    elif mode == "legacy":
        candidates = [fit_data_root]
        note = "legacy artifacts use fit_data root"
    elif mode == "2025":
        candidates = sorted(
            [
                path for path in fit_data_root.iterdir()
                if path.is_dir() and "2025" in path.name and path.name != REPORT_DIR_NAME
            ],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if len(candidates) > 1:
            note = (
                "multiple 2025 directories found; using the most recently modified one"
            )
    elif mode == "6gev":
        sixgev_dir = fit_data_root / "6gev"
        legacy_alias_dir = fit_data_root / "12gev"
        if sixgev_dir.is_dir():
            candidates.append(sixgev_dir)
        if legacy_alias_dir.is_dir():
            candidates.append(legacy_alias_dir)
        if not sixgev_dir.is_dir() and legacy_alias_dir.is_dir():
            note = "6gev artifacts not found; using legacy alias directory 12gev"

    artifact_dir = candidates[0].resolve() if candidates else None
    existing_files = []
    missing_files = []
    status = "missing"

    if artifact_dir is not None:
        existing_files = [
            filename for filename in CANONICAL_STAGE_FILES
            if (artifact_dir / filename).exists()
        ]
        missing_files = [
            filename for filename in CANONICAL_STAGE_FILES
            if not (artifact_dir / filename).exists()
        ]
        if existing_files:
            status = "artifacts_available"
        elif artifact_dir.exists():
            status = "directory_only"
        else:
            status = "missing"

    if status == "missing" and splash_files:
        status = "splash_only"

    return {
        "mode": mode,
        "artifact_dir": artifact_dir,
        "candidate_dirs": candidates,
        "status": status,
        "note": note,
        "splash_files": splash_files,
        "existing_files": existing_files,
        "missing_files": missing_files,
    }


def parse_numeric_sequence(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []

    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(item) for item in np.asarray(value, dtype=float).flatten().tolist()]

    text = str(value).strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].replace("\n", " ").replace(",", " ")
        parsed = np.fromstring(inner, sep=" ")
        if parsed.size:
            return parsed.astype(float).tolist()

    try:
        parsed = ast.literal_eval(text)
    except Exception:
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        return [float(number) for number in numbers]

    if isinstance(parsed, (list, tuple, np.ndarray)):
        return [float(item) for item in np.asarray(parsed, dtype=float).flatten().tolist()]
    if isinstance(parsed, (int, float)):
        return [float(parsed)]
    return []


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, Path):
        return project_display_path(value)
    if isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    return value


def format_sequence(values, precision=6):
    if values is None:
        return "missing"
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return "[]"
    return np.array2string(array, precision=precision, separator=", ", max_line_width=140)


def format_matrix(values, precision=6):
    if values is None:
        return "missing"
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return "[]"
    return np.array2string(array, precision=precision, separator=", ", max_line_width=140)


def normalize_label_text(value):
    return str(value).replace("±", "+/-")


def format_label_counts(label_counts):
    if not label_counts:
        return "none"
    return ", ".join(f"{label}: {count}" for label, count in label_counts.items())


def normalize_discovery_note(note):
    if not note:
        return "none"
    prefix = "user override -> "
    if note.startswith(prefix):
        return f"{prefix}{project_display_path(note[len(prefix):].strip())}"
    return note


def load_bw_summary(artifact_dir):
    csv_path = artifact_dir / "fit_results.csv"
    if not csv_path.exists():
        return {"available": False, "path": csv_path, "parameters": {}}

    dataframe = pd.read_csv(csv_path)
    parameters = {}
    for _, row in dataframe.iterrows():
        parameter = str(row["Parameter"])
        parameters[parameter] = {
            "chi_squared": float(row["Chi-Squared"]),
            "best_fit_parameters": parse_numeric_sequence(row["Best Fit Parameters"]),
            "best_p_values": parse_numeric_sequence(row["Best P Values"]),
            "parameter_uncertainties": parse_numeric_sequence(row["Parameter Uncertainties"]),
            "p_value_uncertainties": parse_numeric_sequence(row["P Value Uncertainties"]),
        }

    return {"available": True, "path": csv_path, "parameters": parameters}


def safe_corrcoef(samples):
    if samples.shape[0] < 2:
        return np.full((samples.shape[1], samples.shape[1]), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.corrcoef(samples, rowvar=False)
    return correlation


def load_bootstrap_summary(artifact_dir):
    bootstrap = {}
    for parameter in BW_PARAMETERS:
        params_path = artifact_dir / f"bootstrap_{parameter}_params.npy"
        if not params_path.exists():
            bootstrap[parameter] = {
                "available": False,
                "path": params_path,
            }
            continue

        samples = np.load(params_path)
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        finite_mask = np.all(np.isfinite(samples), axis=1)
        finite_samples = samples[finite_mask]
        if finite_samples.size == 0:
            covariance = np.array([])
            correlation = np.array([])
            means = np.array([])
            stds = np.array([])
        else:
            covariance = np.cov(finite_samples, rowvar=False)
            correlation = safe_corrcoef(finite_samples)
            means = np.mean(finite_samples, axis=0)
            stds = np.std(finite_samples, axis=0)

        bootstrap[parameter] = {
            "available": True,
            "path": params_path,
            "n_samples": int(samples.shape[0]),
            "n_parameters": int(samples.shape[1]),
            "n_finite_samples": int(finite_samples.shape[0]),
            "mean": means.tolist() if means.size else [],
            "std": stds.tolist() if stds.size else [],
            "covariance": np.asarray(covariance).tolist() if np.asarray(covariance).size else [],
            "correlation": np.asarray(correlation).tolist() if np.asarray(correlation).size else [],
        }
    return bootstrap


def load_transition_summary(artifact_dir):
    results_path = artifact_dir / "full_results.csv"
    errors_path = artifact_dir / "full_results_errors.csv"
    if not results_path.exists() or not errors_path.exists():
        return {
            "available": False,
            "results_path": results_path,
            "errors_path": errors_path,
            "bins": [],
        }

    params_df = pd.read_csv(results_path, index_col=0)
    errors_df = pd.read_csv(errors_path, index_col=0)

    bins = []
    for q2_label in params_df.columns:
        record = {"q2_label": q2_label}
        for parameter in params_df.index:
            value = params_df.at[parameter, q2_label]
            error = errors_df.at[parameter, q2_label] if parameter in errors_df.index else np.nan
            record[parameter] = float(value)
            record[f"{parameter}_err"] = float(error)
        bins.append(record)

    return {
        "available": True,
        "results_path": results_path,
        "errors_path": errors_path,
        "bins": bins,
    }


def load_dis_summary(artifact_dir):
    summary_path = artifact_dir / "dis_fit_summary.json"
    if not summary_path.exists():
        return {"available": False, "path": summary_path}

    with open(summary_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {"available": True, "path": summary_path, "summary": payload}


def gather_mode_results(mode_info):
    artifact_dir = mode_info["artifact_dir"]
    if artifact_dir is None or not artifact_dir.exists():
        return {
            "dis_fit": {"available": False, "path": None},
            "bw_fit": {"available": False, "path": None, "parameters": {}},
            "bootstrap": {parameter: {"available": False, "path": None} for parameter in BW_PARAMETERS},
            "transition_fit": {"available": False, "results_path": None, "errors_path": None, "bins": []},
        }

    return {
        "dis_fit": load_dis_summary(artifact_dir),
        "bw_fit": load_bw_summary(artifact_dir),
        "bootstrap": load_bootstrap_summary(artifact_dir),
        "transition_fit": load_transition_summary(artifact_dir),
    }


def build_discovery_rows(mode_records):
    rows = []
    for record in mode_records:
        artifact_dir = record["artifact_dir"]
        rows.append([
            record["mode"],
            record["status"],
            project_display_path(artifact_dir) if artifact_dir is not None else "missing",
            ", ".join(record["existing_files"]) if record["existing_files"] else "none",
            ", ".join(path.name for path in record["splash_files"]) if record["splash_files"] else "none",
            normalize_discovery_note(record["note"]),
        ])
    return rows


def build_splash_overview_rows(mode_records):
    rows = []
    frame_names = ("g1f1_df", "dis_df", "g2f1_df", "a1_df", "a2_df")
    for record in mode_records:
        splash_summaries = record.get("splash_summaries", [])
        if not splash_summaries:
            rows.append([record["mode"], "missing", "missing", "missing", "missing", "missing", "missing", "missing", "missing"])
            continue

        for splash_summary in splash_summaries:
            row = [
                record["mode"],
                Path(splash_summary["path"]).name,
                splash_summary.get("dataset_mode") or "unknown",
                splash_summary.get("analysis_scope") or "unknown",
            ]
            for frame_name in frame_names:
                row.append(splash_summary["frames"].get(frame_name, {}).get("rows", "missing"))
            rows.append(row)
    return rows


def build_splash_detail_rows(mode_records):
    rows = []
    for record in mode_records:
        splash_summaries = record.get("splash_summaries", [])
        if not splash_summaries:
            rows.append([record["mode"], "missing", "missing", "missing", "missing", "missing", "missing", "missing"])
            continue

        for splash_summary in splash_summaries:
            splash_name = Path(splash_summary["path"]).name
            for frame_name, frame_data in splash_summary["frames"].items():
                rows.append([
                    record["mode"],
                    splash_name,
                    splash_summary.get("analysis_scope") or "unknown",
                    frame_name,
                    frame_data.get("rows", "missing"),
                    frame_data.get("sources_raw", "none"),
                    frame_data.get("cuts_raw", "none"),
                    frame_data.get("labels_raw", format_label_counts(frame_data.get("labels", {}))),
                ])
    return rows


def build_bw_rows(mode_records):
    rows = []
    for record in mode_records:
        bw_fit = record["results"]["bw_fit"]
        if not bw_fit["available"]:
            rows.append([
                record["mode"],
                "missing",
                "n/a",
                "n/a",
                "n/a",
                "n/a",
                "n/a",
            ])
            continue

        for parameter in BW_PARAMETERS:
            parameter_data = bw_fit["parameters"].get(parameter)
            if parameter_data is None:
                rows.append([
                    record["mode"],
                    parameter,
                    "missing",
                    "missing",
                    "missing",
                    "missing",
                    "missing",
                ])
                continue

            rows.append([
                record["mode"],
                parameter,
                f"{parameter_data['chi_squared']:.6g}",
                format_sequence(parameter_data["best_fit_parameters"]),
                format_sequence(parameter_data["parameter_uncertainties"]),
                format_sequence(parameter_data["best_p_values"]),
                format_sequence(parameter_data["p_value_uncertainties"]),
            ])
    return rows


def build_transition_rows(mode_records):
    rows = []
    for record in mode_records:
        transition_fit = record["results"]["transition_fit"]
        if not transition_fit["available"]:
            rows.append([
                record["mode"],
                "missing",
                "n/a",
                "n/a",
            ])
            continue

        for bin_record in transition_fit["bins"]:
            rows.append([
                record["mode"],
                normalize_label_text(bin_record["q2_label"]),
                f"{bin_record['w_dis_transition']:.6g} +/- {bin_record['w_dis_transition_err']:.6g}",
                f"{bin_record['damping_dis_width']:.6g} +/- {bin_record['damping_dis_width_err']:.6g}",
            ])
    return rows


def build_bootstrap_rows(mode_records):
    rows = []
    for record in mode_records:
        for parameter in BW_PARAMETERS:
            bootstrap = record["results"]["bootstrap"][parameter]
            if not bootstrap["available"]:
                rows.append([record["mode"], parameter, "missing", "n/a", "n/a"])
                continue
            rows.append([
                record["mode"],
                parameter,
                bootstrap["n_finite_samples"],
                format_sequence(bootstrap["mean"]),
                format_sequence(bootstrap["std"]),
            ])
    return rows


def build_dis_rows(mode_records):
    rows = []
    for record in mode_records:
        dis_fit = record["results"]["dis_fit"]
        if not dis_fit["available"]:
            rows.append([record["mode"], "missing", "DIS fit summary file not found"])
            continue

        summary = dis_fit["summary"]
        rows.append([
            record["mode"],
            f"{summary.get('chi2_quad', 'n/a')}",
            format_sequence(summary.get("par_quad", [])),
        ])
    return rows


def render_report(summary_payload):
    mode_records = summary_payload["modes"]
    lines = []
    lines.append("=" * 120)
    lines.append("Fit Comparison Report")
    lines.append(f"Generated: {summary_payload['generated_at']}")
    lines.append(f"Fit-data root: {project_display_path(summary_payload['fit_data_root'])}")
    lines.append("")

    lines.append("Mode Discovery")
    lines.append(
        tabulate(
            build_discovery_rows(mode_records),
            headers=["Mode", "Status", "Artifact Dir", "Existing Files", "Splash Files", "Notes"],
            tablefmt=TABLE_FORMAT,
        )
    )
    lines.append("")

    lines.append("DIS Fit Summary")
    lines.append(
        tabulate(
            build_dis_rows(mode_records),
            headers=["Mode", "Chi2", "Parameters / Status"],
            tablefmt=TABLE_FORMAT,
        )
    )
    lines.append("")

    lines.append("DIS Covariance / Correlation Matrices")
    for record in mode_records:
        dis_fit = record["results"]["dis_fit"]
        lines.append(f"Mode: {record['mode']}")
        if not dis_fit["available"]:
            lines.append("  missing")
            lines.append("")
            continue
        summary = dis_fit["summary"]
        lines.append(f"  path: {project_display_path(dis_fit['path'])}")
        lines.append(f"  parameter_names: {', '.join(summary.get('parameter_names', []))}")
        lines.append("  covariance:")
        lines.append(format_matrix(summary.get("cov_quad")))
        lines.append("  correlation:")
        lines.append(format_matrix(summary.get("corr_quad")))
        lines.append("")

    lines.append("BW Global Fit Summary")
    lines.append(
        tabulate(
            build_bw_rows(mode_records),
            headers=[
                "Mode",
                "Parameter",
                "Chi2",
                "Best Fit Parameters",
                "Parameter Uncertainties",
                "Best P Values",
                "P-Value Uncertainties",
            ],
            tablefmt=TABLE_FORMAT,
        )
    )
    lines.append("")

    lines.append("DIS-Transition Bin Summary")
    lines.append(
        tabulate(
            build_transition_rows(mode_records),
            headers=["Mode", "Q2 Label", "w_dis_transition", "damping_dis_width"],
            tablefmt=TABLE_FORMAT,
        )
    )
    lines.append("")

    lines.append("BW Bootstrap Summary")
    lines.append(
        tabulate(
            build_bootstrap_rows(mode_records),
            headers=["Mode", "Parameter", "Finite Samples", "Bootstrap Mean", "Bootstrap Std"],
            tablefmt=TABLE_FORMAT,
        )
    )
    lines.append("")

    lines.append("Bootstrap Covariance / Correlation Matrices")
    for record in mode_records:
        lines.append(f"Mode: {record['mode']}")
        for parameter in BW_PARAMETERS:
            bootstrap = record["results"]["bootstrap"][parameter]
            lines.append(f"  {parameter}:")
            if not bootstrap["available"]:
                lines.append("    missing")
                continue
            lines.append(f"    path: {project_display_path(bootstrap['path'])}")
            lines.append("    covariance:")
            lines.append(format_matrix(bootstrap["covariance"]))
            lines.append("    correlation:")
            lines.append(format_matrix(bootstrap["correlation"]))
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def render_splash_breakdown(summary_payload):
    mode_records = summary_payload["modes"]
    lines = []
    lines.append("=" * 120)
    lines.append("Splash Breakdown Report")
    lines.append(f"Generated: {summary_payload['generated_at']}")
    lines.append(f"Fit-data root: {project_display_path(summary_payload['fit_data_root'])}")
    lines.append("")

    lines.append("Splash Overview")
    lines.append(
        tabulate(
            build_splash_overview_rows(mode_records),
            headers=[
                "Mode",
                "Splash File",
                "Dataset Mode",
                "Scope",
                "g1f1 rows",
                "dis rows",
                "g2f1 rows",
                "a1 rows",
                "a2 rows",
            ],
            tablefmt=TABLE_FORMAT,
        )
    )
    lines.append("")

    lines.append("Splash Frame Details")
    lines.append(
        tabulate(
            build_splash_detail_rows(mode_records),
            headers=[
                "Mode",
                "Splash File",
                "Scope",
                "Frame",
                "Rows",
                "Sources",
                "Cuts",
                "Labels",
            ],
            tablefmt=TABLE_FORMAT,
        )
    )
    lines.append("")

    for record in mode_records:
        splash_summaries = record.get("splash_summaries", [])
        lines.append(f"Mode: {record['mode']}")
        if not splash_summaries:
            lines.append("  missing")
            lines.append("")
            continue

        for splash_summary in splash_summaries:
            lines.append(f"  File: {Path(splash_summary['path']).name}")
            lines.append(
                f"  dataset_mode={splash_summary.get('dataset_mode') or 'unknown'} "
                f"analysis_scope={splash_summary.get('analysis_scope') or 'unknown'}"
            )
            for frame_name, frame_data in splash_summary["frames"].items():
                lines.append(f"    {frame_name}:")
                lines.append(f"      rows: {frame_data.get('rows', 'missing')}")
                lines.append(f"      sources: {frame_data.get('sources_raw', 'none')}")
                lines.append(f"      cuts: {frame_data.get('cuts_raw', 'none')}")
                lines.append(f"      labels: {frame_data.get('labels_raw', format_label_counts(frame_data.get('labels', {})))}")
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def print_console_text(text):
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe_text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(safe_text)


def write_summary_files(output_dir, summary_payload, report_text, splash_breakdown_text):
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "comparison_summary.json"
    report_path = output_dir / "comparison_report.txt"
    splash_report_path = output_dir / "splash_breakdown.txt"
    bw_csv_path = output_dir / "bw_global_summary.csv"
    transition_csv_path = output_dir / "transition_summary.csv"
    bootstrap_csv_path = output_dir / "bootstrap_summary.csv"
    discovery_csv_path = output_dir / "discovery_summary.csv"
    dis_csv_path = output_dir / "dis_fit_summary.csv"
    splash_csv_path = output_dir / "splash_breakdown.csv"

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(sanitize_for_json(summary_payload), handle, indent=2, sort_keys=True)

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report_text)

    with open(splash_report_path, "w", encoding="utf-8") as handle:
        handle.write(splash_breakdown_text)

    pd.DataFrame(
        build_discovery_rows(summary_payload["modes"]),
        columns=["mode", "status", "artifact_dir", "existing_files", "splash_files", "notes"],
    ).to_csv(discovery_csv_path, index=False)

    pd.DataFrame(
        build_dis_rows(summary_payload["modes"]),
        columns=["mode", "chi2", "parameters_or_status"],
    ).to_csv(dis_csv_path, index=False)

    pd.DataFrame(
        build_bw_rows(summary_payload["modes"]),
        columns=[
            "mode",
            "parameter",
            "chi2",
            "best_fit_parameters",
            "parameter_uncertainties",
            "best_p_values",
            "p_value_uncertainties",
        ],
    ).to_csv(bw_csv_path, index=False)

    pd.DataFrame(
        build_transition_rows(summary_payload["modes"]),
        columns=["mode", "q2_label", "w_dis_transition", "damping_dis_width"],
    ).to_csv(transition_csv_path, index=False)

    pd.DataFrame(
        build_bootstrap_rows(summary_payload["modes"]),
        columns=["mode", "parameter", "finite_samples", "bootstrap_mean", "bootstrap_std"],
    ).to_csv(bootstrap_csv_path, index=False)

    pd.DataFrame(
        build_splash_detail_rows(summary_payload["modes"]),
        columns=["mode", "splash_file", "analysis_scope", "frame", "rows", "sources", "cuts", "labels"],
    ).to_csv(splash_csv_path, index=False)

    return {
        "json": json_path,
        "report": report_path,
        "splash_report": splash_report_path,
        "bw_csv": bw_csv_path,
        "transition_csv": transition_csv_path,
        "bootstrap_csv": bootstrap_csv_path,
        "discovery_csv": discovery_csv_path,
        "dis_csv": dis_csv_path,
        "splash_csv": splash_csv_path,
    }


def find_unmapped_tagged_directories(fit_data_root, mode_records):
    mapped_paths = {
        record["artifact_dir"].resolve()
        for record in mode_records
        if record["artifact_dir"] is not None and record["artifact_dir"].exists()
    }
    unmapped = []
    for path in fit_data_root.iterdir():
        if not path.is_dir() or path.name == REPORT_DIR_NAME:
            continue
        if path.resolve() not in mapped_paths:
            unmapped.append(path)
    return sorted(unmapped)


def main():
    args = parse_args()
    fit_data_root = Path(args.fit_data_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else fit_data_root / REPORT_DIR_NAME
    overrides = parse_mode_overrides(args.mode_dir)
    modes = [mode.lower() for mode in args.modes]

    mode_records = []
    for mode in modes:
        mode_info = discover_mode_directory(mode, fit_data_root, overrides)
        mode_info["splash_summaries"] = load_splash_summaries(mode_info["splash_files"])
        mode_info["results"] = gather_mode_results(mode_info)
        mode_records.append(mode_info)

    summary_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "fit_data_root": project_display_path(fit_data_root),
        "modes": mode_records,
        "unmapped_tagged_directories": [
            project_display_path(path)
            for path in find_unmapped_tagged_directories(fit_data_root, mode_records)
        ],
    }

    report_text = render_report(summary_payload)
    splash_breakdown_text = render_splash_breakdown(summary_payload)
    written_files = write_summary_files(output_dir, summary_payload, report_text, splash_breakdown_text)

    print_console_text(report_text)
    if summary_payload["unmapped_tagged_directories"]:
        print("Unmapped tagged directories:")
        for directory in summary_payload["unmapped_tagged_directories"]:
            print(f"  {directory}")
    print("Saved comparison outputs:")
    for label, path in written_files.items():
        print(f"  {label}: {project_display_path(path)}")


if __name__ == "__main__":
    main()
