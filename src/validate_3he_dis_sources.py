#! /usr/bin/python

import math
import sys

import numpy as np

from dis_fit_data_sources import (
    SOURCE_GROUPS,
    build_3he_g1f1_group_bundle,
    load_3he_g1f1_source,
    load_source_manifest,
)


def _fail(message):
    print(f"FAIL {message}")
    raise RuntimeError(message)


def _pass(message):
    print(f"PASS {message}")


def main():
    manifest = load_source_manifest()
    manifest_sources = manifest["sources"]

    for source_key, source_config in manifest_sources.items():
        source_df = load_3he_g1f1_source(source_key, manifest)
        required_columns = {"Q2", "X", "W", "G1F1", "G1F1.err", "Label", "source_key"}
        missing_columns = required_columns.difference(source_df.columns)
        if missing_columns:
            _fail(f"{source_key}: missing canonical columns {sorted(missing_columns)}")
        if not np.isfinite(source_df["Q2"]).all():
            _fail(f"{source_key}: non-finite Q2 values")
        if not np.isfinite(source_df["X"]).all():
            _fail(f"{source_key}: non-finite X values")
        if not np.isfinite(source_df["G1F1"]).all():
            _fail(f"{source_key}: non-finite G1F1 values")
        if not np.isfinite(source_df["G1F1.err"]).all():
            _fail(f"{source_key}: non-finite G1F1.err values")
        if not (source_df["G1F1.err"] > 0).all():
            _fail(f"{source_key}: non-positive G1F1.err values")
        _pass(f"{source_key}: loaded {len(source_df)} rows from {source_config['file']}")

    hermes_df = load_3he_g1f1_source("hermes_2000", manifest)
    if len(hermes_df) != 9:
        _fail(f"hermes_2000: expected 9 rows, observed {len(hermes_df)}")
    _pass("hermes_2000: loaded exactly 9 rows")

    zheng_df = load_3he_g1f1_source("e99117_zheng", manifest)
    if len(zheng_df) != 3:
        _fail(f"e99117_zheng: expected 3 rows, observed {len(zheng_df)}")
    _pass("e99117_zheng: loaded exactly 3 rows")

    flay_df = load_3he_g1f1_source("e06014_flay", manifest)
    if len(flay_df) < 1:
        _fail("e06014_flay: no rows loaded before DIS cuts")
    _pass(f"e06014_flay: loaded {len(flay_df)} rows before DIS cuts")

    current_bundle = build_3he_g1f1_group_bundle(
        "current_global_2025",
        manifest,
        source_groups=SOURCE_GROUPS,
        q2_min=1.0,
        dis_w_min=2.0,
    )
    current_keys = set(current_bundle["metadata"]["source_keys"])
    if "a1n_2025_dis" not in current_keys or "mingyu_legacy_dis" in current_keys:
        _fail("current_global_2025: expected a1n_2025_dis and excluded mingyu_legacy_dis")
    if "hermes_2000" not in current_keys:
        _fail("current_global_2025: expected hermes_2000")
    _pass("current_global_2025: includes 2025 + HERMES and excludes Mingyu")

    legacy_bundle = build_3he_g1f1_group_bundle(
        "legacy_mingyu",
        manifest,
        source_groups=SOURCE_GROUPS,
        q2_min=1.0,
        dis_w_min=2.0,
    )
    legacy_keys = set(legacy_bundle["metadata"]["source_keys"])
    if "mingyu_legacy_dis" not in legacy_keys or "a1n_2025_dis" in legacy_keys:
        _fail("legacy_mingyu: expected mingyu_legacy_dis and excluded a1n_2025_dis")
    _pass("legacy_mingyu: includes Mingyu and excludes 2025")

    full_current_bundle = build_3he_g1f1_group_bundle(
        "full_current_global_2025",
        manifest,
        source_groups=SOURCE_GROUPS,
        q2_min=1.0,
        dis_w_min=2.0,
    )
    full_current_keys = set(full_current_bundle["metadata"]["source_keys"])
    for support_key in ("e94010", "e97110"):
        if support_key not in full_current_keys:
            _fail(f"full_current_global_2025: missing full-analysis support source {support_key}")
    _pass("full_current_global_2025: includes E94-010 and E97-110 support sources")

    flay_audit_row = next(
        row for row in full_current_bundle["metadata"]["audit_rows"]
        if row["source_key"] == "e06014_flay"
    )
    if flay_audit_row["N_loaded"] < flay_audit_row["N_dis"]:
        _fail("e06014_flay: DIS-cut rows exceeded loaded rows")
    _pass(
        f"e06014_flay: preserved {flay_audit_row['N_loaded']} raw rows before DIS cuts and "
        f"{flay_audit_row['removed_by_dis_cut']} were removed by the DIS cut"
    )

    print("All source-aware 3He DIS validation checks passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Validation failed: {exc}")
        raise SystemExit(1)
