# 3He DIS Source Inventory

This document describes the source-aware 3He `g1/F1` input system used by the opt-in `DIS_DATA_MODE="source_group"` path.

## Policy
- `g1f1_comb.csv` remains the legacy combined-table path and is still the default.
- The source-aware path is only for 3He `g1/F1` in this phase.
- `g2f1`, `a1`, and `a2` remain on the current legacy-combined loaders.
- Do not double count raw source files with `g1f1_comb.csv`.
- Do not mix `a1n_2025_dis` and `mingyu_legacy_dis` in the same nominal source group.
- `a1n_2025_all` is diagnostic-only unless a diagnostic source group explicitly selects it.

## Source Table

| Source key | Label | File | Provenance | Used by default? | Notes |
|---|---|---|---|---|---|
| `e142` | `SLAC E142 (1996)` | `data/slac_e142.csv` | `PRD54(1996)6620; hep-ex/9610007; Table 7` | yes in source-aware groups | Uses `G1F1.cal` and `G1F1.cal.err` |
| `e154_yury` | `SLAC E154 / Yury` | `data/slac_e154.csv` | `Yury Kolomensky thesis and private communication` | yes in source-aware groups | Preserve Yury/private-communication provenance in audits |
| `hermes_2000` | `HERMES (2000)` | `data/hermes_2000.csv` | `hep-ex/9906035; Table V inclusive 3He top block` | yes in current global groups | Treat `A1He3.mes` as a `g1/F1` proxy and recompute `W` |
| `e99117_zheng` | `Zheng E99-117 (2002)` | `data/zheng_thesis_pub_e99117.csv` | `PRC / nucl-ex/0405006` | yes in source-aware groups | Uses published `G1F1.mes` columns |
| `e06014_flay` | `Flay E06-014 (2014)` | `data/dflay_e06014.csv` | `David Flay long PRD Tables VI and VII` | yes in source-aware groups | DIS + RES, no source-level W cut |
| `e97103_kramer` | `Kramer E97-103 (2003)` | `data/kramer_e97103.csv` | Existing processed local plot source | group-controlled | Included only when the selected source group says so |
| `a1n_2025_dis` | `A1n 2025 DIS` | `data/g1F1he3_2025_dis.csv` | Current internal 2025 extraction | yes in current 2025 groups | Nominal current internal DIS source |
| `a1n_2025_all` | `A1n 2025 all` | `data/g1F1he3_2025_all.csv` | Current broader 2025 extraction | no | Diagnostic-only source group member |
| `mingyu_legacy_dis` | `Mingyu DIS legacy` | `data/mingyu_g1f1_g2f1_dis.csv` | Legacy internal extraction | no | Legacy-only source |
| `e94010` | `E94-010` | `data/e94010.csv` | Historical processed resonance-support table | full groups only | Added to `full_*` source groups for BW/resonance coverage |
| `e97110` | `E97-110` | `data/e97110.csv` | Historical processed resonance-support table | full groups only | Added to `full_*` source groups for BW/resonance coverage |

## Observable Mapping
- `e142`, `e154_yury`, `e97103_kramer`: `Q2 <- Q2`, `X <- X`, `W <- W.cal`, `G1F1 <- G1F1.cal`, `G1F1.err <- G1F1.cal.err`
- `e99117_zheng`: `Q2 <- Q2`, `X <- X`, `W <- W`, `G1F1 <- G1F1.mes`, `G1F1.err <- G1F1.mes.err`
- `e06014_flay`: `Q2 <- Q2`, `X <- X`, `W <- W`, `G1F1 <- G1F1.mes`, `G1F1.err <- G1F1.err.total`
- `hermes_2000`: `Q2 <- Q2`, `X <- X`, `G1F1 <- A1He3.mes`, `G1F1.err <- A1He3.mes.err`, `W` recomputed from `X` and `Q2`
- `a1n_2025_dis`, `a1n_2025_all`: `Q2 <- Q2`, `X <- xbj`, `G1F1 <- g1/F1_He3`, `G1F1.err <- sqrt(stat^2 + syst^2)`, `W` recomputed from `X` and `Q2`
- `mingyu_legacy_dis`: `Q2 <- Q2`, `X <- x`, `W <- W.cal`, `G1F1 <- g1F1_3He`, `G1F1.err <- g1f1.err`
- `e94010`, `e97110`: `Q2 <- Q2`, `X <- X`, `W <- W`, `G1F1 <- G1F1.cal`, `G1F1.err <- G1F1.cal.err`

## Source Groups
- `plots_baseline`: `e142`, `e154_yury`, `e99117_zheng`, `e06014_flay`, `e97103_kramer`
- `plots_baseline_plus_hermes`: baseline plus `hermes_2000`
- `current_global_2025`: baseline plus `hermes_2000` and `a1n_2025_dis`
- `current_global_2025_no_kramer`: current 2025 group without Kramer
- `legacy_mingyu`: baseline plus `hermes_2000` and `mingyu_legacy_dis`
- `current_2025_all_diagnostic`: baseline plus `hermes_2000` and `a1n_2025_all`
- Any `full_<group>` companion adds `e94010` and `e97110`

## Operational Notes
- In source-aware mode, `dis_df` applies the fit-level DIS cut after source assembly: `Q2 > 1.0` and optionally `W > DIS_W_MIN`.
- `g1f1_df` in source-aware mode keeps the raw loaded rows and never applies source-level W cuts.
- `fit_data/<analysis_tag>/dis_fit_source_breakdown.txt` and `dis_fit_source_audit.csv` are the authoritative per-run source audit files.
