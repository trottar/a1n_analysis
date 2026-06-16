#!/usr/bin/env python3
"""
Filter A1n plotted CSVs into cleaner summary CSVs.

Default behavior:
  - drops estimator = hist_ratio_bin
  - keeps direct plotted rows, but relabels blank estimator as direct_mean
  - keeps estimator = hist_ratio_mean

Modes:
  --xsec-only    keeps only estimator = hist_ratio_mean
  --direct-only  keeps only direct plotted rows
  --drop-empty   drops rows where value is NaN/blank

Examples:
  python3 filter_a1n_plotted_csv_v2.py \
    OUTPUT/compare/compare_kin_datasimc_hms_cuts1-3-5-6-7-8_plotted.csv

  python3 filter_a1n_plotted_csv_v2.py \
    OUTPUT/compare/compare_kin_datasimc_hms_cuts1-3-5-6-7-8_plotted.csv \
    --xsec-only

  python3 filter_a1n_plotted_csv_v2.py input.csv --xsec-only --drop-empty -o xsec_summary.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def default_output_path(input_path: Path, suffix: str) -> Path:
    stem = input_path.stem
    if stem.endswith("_plotted"):
        stem = stem[: -len("_plotted")]
    return input_path.with_name(f"{stem}_{suffix}.csv")


def filter_plotted_csv(
    input_csv: Path,
    output_csv: Path,
    xsec_only: bool = False,
    direct_only: bool = False,
    keep_hist_bins: bool = False,
    drop_empty: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    if "estimator" not in df.columns:
        df["estimator"] = "direct_mean"

    estimator = df["estimator"].fillna("").astype(str).str.strip()

    direct_mask = estimator.eq("")
    xsec_mean_mask = estimator.eq("hist_ratio_mean")
    hist_bin_mask = estimator.eq("hist_ratio_bin")

    # Relabel blank direct rows so the output is no longer ambiguous.
    df.loc[direct_mask, "estimator"] = "direct_mean"
    estimator = df["estimator"].fillna("").astype(str).str.strip()
    direct_mask = estimator.eq("direct_mean")

    if xsec_only:
        keep_mask = xsec_mean_mask
    elif direct_only:
        keep_mask = direct_mask
    elif keep_hist_bins:
        keep_mask = pd.Series(True, index=df.index)
    else:
        keep_mask = direct_mask | xsec_mean_mask

    if not keep_hist_bins:
        keep_mask &= ~hist_bin_mask

    out = df.loc[keep_mask].copy()

    if drop_empty and "value" in out.columns:
        out = out[pd.to_numeric(out["value"], errors="coerce").notna()].copy()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    print(f"[input]  {input_csv}")
    print(f"[output] {output_csv}")
    print(f"[rows]   {len(df)} -> {len(out)}")

    print("\n[estimator counts before]")
    print(df["estimator"].fillna("<blank>").value_counts(dropna=False).to_string())

    print("\n[estimator counts after]")
    print(out["estimator"].fillna("<blank>").value_counts(dropna=False).to_string())

    return out


def parse_args():
    ap = argparse.ArgumentParser(description="Create cleaner summary CSVs from A1n plotted CSVs.")
    ap.add_argument("input_csv", type=Path, help="Input plotted CSV")
    ap.add_argument("-o", "--output", type=Path, default=None, help="Output CSV path")
    ap.add_argument("--xsec-only", action="store_true", help="Keep only estimator=hist_ratio_mean rows")
    ap.add_argument("--direct-only", action="store_true", help="Keep only direct plotted rows")
    ap.add_argument("--keep-hist-bins", action="store_true", help="Do not drop estimator=hist_ratio_bin rows")
    ap.add_argument("--drop-empty", action="store_true", help="Drop rows where value is blank/NaN")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.xsec_only and args.direct_only:
        raise SystemExit("[error] --xsec-only and --direct-only cannot both be used")

    if args.output is None:
        if args.xsec_only:
            suffix = "xsec_summary"
        elif args.direct_only:
            suffix = "direct_summary"
        else:
            suffix = "summary"
        args.output = default_output_path(args.input_csv, suffix)

    filter_plotted_csv(
        input_csv=args.input_csv,
        output_csv=args.output,
        xsec_only=args.xsec_only,
        direct_only=args.direct_only,
        keep_hist_bins=args.keep_hist_bins,
        drop_empty=args.drop_empty,
    )


if __name__ == "__main__":
    main()
