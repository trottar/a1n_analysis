#! /usr/bin/python

import json
import os

import numpy as np
import pandas as pd

from utility import project_display_path, project_path

MANIFEST_FILENAME = "source_manifest_3he_dis.json"
DEFAULT_SOURCE_GROUP = "full_current_global_2025"
FULL_ANALYSIS_SUPPORT_SOURCE_KEYS = ("e94010", "e97110")
CANONICAL_COLUMNS = [
    "source_key",
    "source_group",
    "Label",
    "Q2",
    "X",
    "W",
    "G1F1",
    "G1F1.err",
    "reference",
    "table",
    "notes",
    "is_external",
    "is_internal",
    "is_legacy",
]
TWENTY25_COLUMNS = [
    "Ep", "xbj", "Q2",
    "Apar", "Apar_stat", "Apar_syst",
    "Aperp", "Aperp_stat", "Aperp_syst",
    "vpar", "vperp",
    "g1F1_He3", "g1F1_stat", "g1F1_syst",
    "vpar2", "vperp2",
    "g2F1_He3", "g2F1_stat", "g2F1_syst",
]
SOURCE_GROUPS = {
    "plots_baseline": [
        "e142",
        "e154_yury",
        "e99117_zheng",
        "e06014_flay",
        "e97103_kramer",
    ],
    "plots_baseline_plus_hermes": [
        "e142",
        "e154_yury",
        "e99117_zheng",
        "e06014_flay",
        "e97103_kramer",
        "hermes_2000",
    ],
    "current_global_2025": [
        "e142",
        "e154_yury",
        "e99117_zheng",
        "e06014_flay",
        "e97103_kramer",
        "hermes_2000",
        "a1n_2025_dis",
    ],
    "current_global_2025_no_kramer": [
        "e142",
        "e154_yury",
        "e99117_zheng",
        "e06014_flay",
        "hermes_2000",
        "a1n_2025_dis",
    ],
    "legacy_mingyu": [
        "e142",
        "e154_yury",
        "e99117_zheng",
        "e06014_flay",
        "e97103_kramer",
        "hermes_2000",
        "mingyu_legacy_dis",
    ],
    "current_2025_all_diagnostic": [
        "e142",
        "e154_yury",
        "e99117_zheng",
        "e06014_flay",
        "e97103_kramer",
        "hermes_2000",
        "a1n_2025_all",
    ],
}
for group_name, source_keys in list(SOURCE_GROUPS.items()):
    full_group_name = f"full_{group_name}"
    SOURCE_GROUPS[full_group_name] = list(source_keys) + list(FULL_ANALYSIS_SUPPORT_SOURCE_KEYS)


def manifest_path():
    return project_path("data", MANIFEST_FILENAME)


def load_source_manifest(path=None):
    manifest_file = path or manifest_path()
    with open(manifest_file, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if "sources" not in manifest:
        raise ValueError("3He DIS source manifest is missing the top-level 'sources' mapping.")
    return manifest


def recompute_w_from_x_q2(x, q2, mass=0.9382720813):
    x_array = np.asarray(x, dtype=np.double)
    q2_array = np.asarray(q2, dtype=np.double)
    with np.errstate(divide="ignore", invalid="ignore"):
        w2 = mass ** 2 + q2_array * (1.0 / x_array - 1.0)
    w2 = np.where(w2 >= 0.0, w2, np.nan)
    return np.sqrt(w2)


def get_source_groups():
    return {key: list(value) for key, value in SOURCE_GROUPS.items()}


def get_source_group_source_keys(group_name, source_groups=None):
    resolved_source_groups = source_groups or SOURCE_GROUPS
    if group_name not in resolved_source_groups:
        supported = ", ".join(sorted(resolved_source_groups))
        raise ValueError(
            f"Unsupported DIS source group '{group_name}'. Expected one of: {supported}."
        )
    return list(resolved_source_groups[group_name])


def describe_source_group(group_name, source_groups=None):
    return ", ".join(get_source_group_source_keys(group_name, source_groups=source_groups))


def _strip_columns(df):
    stripped = df.copy()
    stripped.columns = [str(column).strip() for column in stripped.columns]
    return stripped


def _resolve_source_path(relative_path):
    normalized = os.path.normpath(relative_path)
    return project_path(*normalized.split(os.sep))


def _parser_is_2025(source_config):
    return source_config.get("parser") == "a1n_2025_whitespace"


def _load_2025_source_frame(path):
    raw_df = pd.read_csv(
        path,
        sep=r"\s+",
        names=TWENTY25_COLUMNS,
        skiprows=1,
        engine="python",
    )
    for column in TWENTY25_COLUMNS:
        raw_df[column] = pd.to_numeric(raw_df[column], errors="coerce")
    return raw_df


def _normalize_boolean_source_kind(source_config, key):
    kind = str(source_config.get("kind", "")).strip().lower()
    if key == "is_external":
        return kind == "external"
    if key == "is_internal":
        return kind == "internal"
    if key == "is_legacy":
        return bool(source_config.get("legacy", False))
    raise KeyError(f"Unsupported boolean kind key '{key}'.")


def _build_standard_canonical_frame(raw_df, source_key, source_group, source_config):
    column_map = source_config["columns"]
    required_columns = [column_map["Q2"], column_map["X"], column_map["G1F1"], column_map["G1F1.err"]]
    for required_column in required_columns:
        if required_column not in raw_df.columns:
            raise KeyError(
                f"Source '{source_key}' is missing required column '{required_column}' in "
                f"{project_display_path(_resolve_source_path(source_config['file']))}."
            )

    q2 = pd.to_numeric(raw_df[column_map["Q2"]], errors="coerce")
    x = pd.to_numeric(raw_df[column_map["X"]], errors="coerce")
    g1f1 = pd.to_numeric(raw_df[column_map["G1F1"]], errors="coerce")
    g1f1_err = pd.to_numeric(raw_df[column_map["G1F1.err"]], errors="coerce")

    w_column = column_map.get("W")
    use_existing_w = (
        bool(w_column)
        and w_column in raw_df.columns
        and not bool(source_config.get("recompute_W", False))
    )
    if use_existing_w:
        w_values = pd.to_numeric(raw_df[w_column], errors="coerce")
        if int(w_values.notna().sum()) == 0:
            use_existing_w = False
            w_values = recompute_w_from_x_q2(x, q2)
    else:
        w_values = recompute_w_from_x_q2(x, q2)

    canonical_df = pd.DataFrame(
        {
            "source_key": source_key,
            "source_group": source_group,
            "Label": source_config["label"],
            "Q2": q2,
            "X": x,
            "W": w_values,
            "G1F1": g1f1,
            "G1F1.err": g1f1_err,
            "reference": source_config.get("reference", ""),
            "table": source_config.get("table", ""),
            "notes": source_config.get("notes", ""),
            "is_external": _normalize_boolean_source_kind(source_config, "is_external"),
            "is_internal": _normalize_boolean_source_kind(source_config, "is_internal"),
            "is_legacy": _normalize_boolean_source_kind(source_config, "is_legacy"),
        }
    )
    metadata = {
        "source_path": _resolve_source_path(source_config["file"]),
        "w_recomputed": not use_existing_w,
        "parser": source_config.get("parser", "csv"),
        "raw_rows": int(len(raw_df)),
    }
    return canonical_df, metadata


def _build_2025_canonical_frame(raw_df, source_key, source_group, source_config):
    q2 = pd.to_numeric(raw_df["Q2"], errors="coerce")
    x = pd.to_numeric(raw_df["xbj"], errors="coerce")
    g1f1 = pd.to_numeric(raw_df["g1F1_He3"], errors="coerce")
    stat = pd.to_numeric(raw_df["g1F1_stat"], errors="coerce")
    syst = pd.to_numeric(raw_df["g1F1_syst"], errors="coerce")
    g1f1_err = np.sqrt(stat ** 2 + syst ** 2)
    w_values = recompute_w_from_x_q2(x, q2)

    canonical_df = pd.DataFrame(
        {
            "source_key": source_key,
            "source_group": source_group,
            "Label": source_config["label"],
            "Q2": q2,
            "X": x,
            "W": w_values,
            "G1F1": g1f1,
            "G1F1.err": g1f1_err,
            "reference": source_config.get("reference", ""),
            "table": source_config.get("table", ""),
            "notes": source_config.get("notes", ""),
            "is_external": _normalize_boolean_source_kind(source_config, "is_external"),
            "is_internal": _normalize_boolean_source_kind(source_config, "is_internal"),
            "is_legacy": _normalize_boolean_source_kind(source_config, "is_legacy"),
        }
    )
    metadata = {
        "source_path": _resolve_source_path(source_config["file"]),
        "w_recomputed": True,
        "parser": source_config.get("parser", "a1n_2025_whitespace"),
        "raw_rows": int(len(raw_df)),
    }
    return canonical_df, metadata


def load_3he_g1f1_source(source_key, manifest, source_group="ungrouped"):
    manifest_sources = manifest["sources"]
    if source_key not in manifest_sources:
        supported = ", ".join(sorted(manifest_sources))
        raise ValueError(f"Unknown source key '{source_key}'. Expected one of: {supported}.")

    source_config = manifest_sources[source_key]
    source_path = _resolve_source_path(source_config["file"])
    if _parser_is_2025(source_config):
        raw_df = _load_2025_source_frame(source_path)
        canonical_df, metadata = _build_2025_canonical_frame(raw_df, source_key, source_group, source_config)
    else:
        raw_df = _strip_columns(pd.read_csv(source_path))
        canonical_df, metadata = _build_standard_canonical_frame(raw_df, source_key, source_group, source_config)

    canonical_df = canonical_df.replace([np.inf, -np.inf], np.nan)
    canonical_df = canonical_df.dropna(subset=["Q2", "X", "W", "G1F1", "G1F1.err"]).copy()
    canonical_df["Q2"] = pd.to_numeric(canonical_df["Q2"], errors="coerce")
    canonical_df["X"] = pd.to_numeric(canonical_df["X"], errors="coerce")
    canonical_df["W"] = pd.to_numeric(canonical_df["W"], errors="coerce")
    canonical_df["G1F1"] = pd.to_numeric(canonical_df["G1F1"], errors="coerce")
    canonical_df["G1F1.err"] = pd.to_numeric(canonical_df["G1F1.err"], errors="coerce")
    canonical_df = canonical_df[canonical_df["G1F1.err"] > 0].copy()
    canonical_df = canonical_df.reset_index(drop=True)
    canonical_df = canonical_df[CANONICAL_COLUMNS]
    metadata["loaded_rows"] = int(len(canonical_df))
    canonical_df.attrs["source_metadata"] = metadata
    return canonical_df


def _apply_dis_cut(df, q2_min=None, dis_w_min=None):
    selection = pd.Series(True, index=df.index)
    cut_descriptions = []
    if q2_min is not None:
        selection &= df["Q2"] > float(q2_min)
        cut_descriptions.append(f"Q2 > {float(q2_min):.3f}")
    if dis_w_min is not None:
        selection &= df["W"] > float(dis_w_min)
        cut_descriptions.append(f"W > {float(dis_w_min):.3f}")
    selected_df = df.loc[selection].copy().reset_index(drop=True)
    return selected_df, cut_descriptions


def build_3he_g1f1_group_bundle(group_name, manifest, source_groups=None, *, dis_w_min=None, q2_min=None):
    resolved_source_groups = source_groups or SOURCE_GROUPS
    source_keys = get_source_group_source_keys(group_name, source_groups=resolved_source_groups)

    full_frames = []
    dis_frames = []
    audit_rows = []
    cuts_applied = []
    recomputed_w_sources = []
    source_file_map = {}
    source_labels = {}

    for source_key in source_keys:
        source_df = load_3he_g1f1_source(source_key, manifest, source_group=group_name)
        source_metadata = dict(source_df.attrs.get("source_metadata", {}))
        full_frames.append(source_df)

        dis_source_df, cut_descriptions = _apply_dis_cut(
            source_df,
            q2_min=q2_min,
            dis_w_min=dis_w_min,
        )
        dis_frames.append(dis_source_df)

        if cut_descriptions:
            cuts_applied.append(f"{source_key}: {', '.join(cut_descriptions)}")
        if source_metadata.get("w_recomputed"):
            recomputed_w_sources.append(source_key)

        source_file_map[source_key] = project_display_path(source_metadata.get("source_path", ""))
        source_labels[source_key] = source_df["Label"].iloc[0] if not source_df.empty else manifest["sources"][source_key]["label"]

        audit_rows.append(
            {
                "source_key": source_key,
                "Label": source_labels[source_key],
                "N_raw": int(source_metadata.get("raw_rows", len(source_df))),
                "N_loaded": int(source_metadata.get("loaded_rows", len(source_df))),
                "N_dis": int(len(dis_source_df)),
                "removed_by_dis_cut": int(len(source_df) - len(dis_source_df)),
                "x_min": float(source_df["X"].min()) if not source_df.empty else np.nan,
                "x_max": float(source_df["X"].max()) if not source_df.empty else np.nan,
                "Q2_min": float(source_df["Q2"].min()) if not source_df.empty else np.nan,
                "Q2_max": float(source_df["Q2"].max()) if not source_df.empty else np.nan,
                "W_min": float(source_df["W"].min()) if not source_df.empty else np.nan,
                "W_max": float(source_df["W"].max()) if not source_df.empty else np.nan,
                "G1F1_min": float(source_df["G1F1"].min()) if not source_df.empty else np.nan,
                "G1F1_max": float(source_df["G1F1"].max()) if not source_df.empty else np.nan,
                "mean_err": float(source_df["G1F1.err"].mean()) if not source_df.empty else np.nan,
                "reference": source_df["reference"].iloc[0] if not source_df.empty else manifest["sources"][source_key].get("reference", ""),
                "table": source_df["table"].iloc[0] if not source_df.empty else manifest["sources"][source_key].get("table", ""),
                "notes": source_df["notes"].iloc[0] if not source_df.empty else manifest["sources"][source_key].get("notes", ""),
                "w_recomputed": bool(source_metadata.get("w_recomputed", False)),
                "source_path": source_file_map[source_key],
            }
        )

    g1f1_df = pd.concat(full_frames, ignore_index=True) if full_frames else pd.DataFrame(columns=CANONICAL_COLUMNS)
    dis_df = pd.concat(dis_frames, ignore_index=True) if dis_frames else pd.DataFrame(columns=CANONICAL_COLUMNS)

    metadata = {
        "source_group": group_name,
        "source_keys": source_keys,
        "source_labels": source_labels,
        "source_file_map": source_file_map,
        "cuts_applied": cuts_applied,
        "recomputed_w_sources": recomputed_w_sources,
        "audit_rows": audit_rows,
        "q2_min": q2_min,
        "dis_w_min": dis_w_min,
    }
    g1f1_df.attrs["source_group_metadata"] = metadata
    dis_df.attrs["source_group_metadata"] = metadata
    return {
        "g1f1_df": g1f1_df.reset_index(drop=True),
        "dis_df": dis_df.reset_index(drop=True),
        "metadata": metadata,
    }


def load_3he_g1f1_group(group_name, manifest, source_groups, *, dis_w_min=None, q2_min=None, apply_dis_cut=False):
    bundle = build_3he_g1f1_group_bundle(
        group_name,
        manifest,
        source_groups=source_groups,
        dis_w_min=dis_w_min,
        q2_min=q2_min,
    )
    if apply_dis_cut:
        return bundle["dis_df"]
    return bundle["g1f1_df"]


def source_group_breakdown_lines(metadata):
    group_name = metadata["source_group"]
    lines = [
        "=" * 100,
        f"[source_group] active group={group_name}",
        f"[source_group] source keys={', '.join(metadata['source_keys'])}",
        f"[source_group] W recomputed for: {', '.join(metadata['recomputed_w_sources']) if metadata['recomputed_w_sources'] else 'none'}",
        f"[source_group] DIS cuts: {', '.join(metadata['cuts_applied']) if metadata['cuts_applied'] else 'none'}",
        "[source_group] per-source counts:",
    ]
    for row in metadata["audit_rows"]:
        lines.append(
            "  "
            + f"{row['source_key']} ({row['Label']}): "
            + f"loaded={row['N_loaded']}, dis={row['N_dis']}, removed_by_dis_cut={row['removed_by_dis_cut']}, "
            + f"file={row['source_path']}"
        )
    lines.append("=" * 100)
    return lines


def source_group_audit_frame(metadata):
    return pd.DataFrame(metadata["audit_rows"])


def write_source_group_reports(metadata, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    breakdown_path = os.path.join(output_dir, "dis_fit_source_breakdown.txt")
    audit_path = os.path.join(output_dir, "dis_fit_source_audit.csv")

    breakdown_lines = source_group_breakdown_lines(metadata)
    with open(breakdown_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(breakdown_lines) + "\n")

    audit_df = source_group_audit_frame(metadata)
    audit_df.to_csv(audit_path, index=False)
    return breakdown_path, audit_path, breakdown_lines
