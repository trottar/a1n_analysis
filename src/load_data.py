#! /usr/bin/python

#
# Description:
# ================================================================
# Time-stamp: "2025-04-22 10:53:24 trottar"
# ================================================================
#
# Author:  Richard L. Trotta III <trottar.iii@gmail.com>
#
# Copyright (c) trottar
#
import pandas as pd
import numpy as np
##################################################################################################################################################

def load_data():

    # # fix Mingyu W.cal
    # # mingyu_df = mingyu_df.drop(columns=["W"])
    # mingyu_df["W.cal"] = W_cal(mingyu_df["Q2"], mingyu_df["x"])
    # mingyu_df.head(10)
    # mingyu_df.to_csv(dir + 'mingyu_g1f1_g2f1_dis.csv', index=False)


    # Load csv files into data frames
    dir = '../data/'
    e06014_df = pd.read_csv(dir + 'dflay_e06014.csv')
    e94010_df = pd.read_csv(dir + 'e94010.csv')
    e97110_df = pd.read_csv(dir + 'e97110.csv')
    psolva1a2_df = pd.read_csv(dir + 'psolv_e01012_a1a2.csv')
    psolvg1g2_df = pd.read_csv(dir + 'psolv_e01012_g1g2.csv')
    zheng_df = pd.read_csv(dir + 'zheng_thesis_pub_e99117.csv')
    hermes_df = pd.read_csv(dir + 'hermes_2000.csv')
    e142_df = pd.read_csv(dir + 'slac_e142.csv')
    e154_df = pd.read_csv(dir + 'slac_e154.csv')
    e97103_df = pd.read_csv(dir + 'kramer_e97103.csv')

    mingyu_df = pd.read_csv(dir + 'mingyu_g1f1_g2f1_dis.csv') # mingyu thesis DIS


    # Saikat's data tables for interpolation
    # caldata = pd.read_csv(dir + 'saikat_tables/XZ_table_3He_JAM_smeared_kpsv_onshell_ipol1_ipolres1_IA14_SF23_AC11.csv') #  0.1<Q2<15.0 GeV2
    # caldata = pd.read_csv(dir + 'saikat_tables/table_3He_JAM_smeared_kpsv_onshell_ipol1_ipolres1_IA14_SF23_AC11.csv') #  0.001<Q2<5.0 GeV2

    # combined g1f1, g2f1, a1, a2 tables
    g1f1_df = pd.read_csv(dir + 'g1f1_comb.csv')
    g2f1_df = pd.read_csv(dir + 'g2f1_comb.csv')
    a1_df = pd.read_csv(dir + 'a1_comb.csv')
    a2_df = pd.read_csv(dir + 'a2_comb.csv')
    
    dis_df = g1f1_df
    
    # --- Debug: Print column names and a sample of data to check the format ---
    print("Columns:", g1f1_df.columns.tolist())

    # --- Function to convert Q² values to float safely, removing any extra characters ---
    def convert_q2(q2):
        try:
            return float(q2)
        except Exception:
            match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(q2))
            if match:
                return float(match.group())
            return np.nan

    # Clean the Q2 column
    g1f1_df['Q2'] = g1f1_df['Q2'].apply(convert_q2)

    # Debug: Print unique Q2 values after cleaning
    unique_q2 = sorted(g1f1_df['Q2'].dropna().unique())
    print("\nUnique Q2 values after cleaning:", unique_q2)

    # --- Remove unwanted experiments ---
    g1f1_df = g1f1_df[~g1f1_df['Label'].isin(["Flay E06-014 (2014)", "Kramer E97-103 (2003)"])]

    # --- Assign Q² category based on the numeric Q² value ---
    def assign_q2_category(q2):
        if pd.isna(q2):
            return None
        if q2 < 0.1:
            return "Low Q2"
        elif 0.1 <= q2 < 1.0:
            return "Mid Q2"
        elif q2 >= 1.0:
            return "High Q2"
        else:
            return None

    g1f1_df['Q2_category'] = g1f1_df['Q2'].apply(assign_q2_category)

    # Debug: Print distribution of Q² categories
    print("\nDistribution of Q² categories:")
    print(g1f1_df['Q2_category'].value_counts(dropna=False))

    # --- New function to create bins with as many points as possible per bin,
    #     but ensuring every bin has at least min_count data points.
    def create_bins_for_category_maximize(df, category, min_count=5, gap_factor=2.0):
        """
        For the given category, this function:
          1. Sorts the Q² values.
          2. Computes the differences between consecutive Q² values.
          3. Flags potential splits when a gap exceeds (gap_factor * median_gap).
          4. Splits the data at those indices.
          5. Merges any bins that have fewer than min_count data points.
        The final label for each bin shows the central value and the bin size.
        """
        subset = df[df['Q2_category'] == category].copy()
        if subset.empty:
            return pd.Series(dtype=object)

        subset.sort_values('Q2', inplace=True)
        q2_vals = subset['Q2'].values
        n = len(q2_vals)
        indices = subset.index.tolist()

        if n < min_count:
            # Not enough points even for one bin; return one bin.
            center = (q2_vals[0] + q2_vals[-1]) / 2
            bin_size = q2_vals[-1] - q2_vals[0]
            label = f"{category} bin 1: {center:.3f} ± {bin_size:.3f} (n={n})"
            return pd.Series([label] * n, index=indices)

        # Compute gaps between consecutive Q² values.
        diffs = np.diff(q2_vals)
        median_diff = np.median(diffs) if len(diffs) > 0 else 0
        threshold = gap_factor * median_diff

        # Identify indices where the gap is "large"
        potential_splits = [i for i, d in enumerate(diffs) if d > threshold]

        # Create initial bins using these split indices.
        bins = []
        start = 0
        for split_idx in potential_splits:
            end = split_idx + 1
            bins.append((start, end))
            start = end
        bins.append((start, n))

        # Merge any bins that have fewer than min_count points.
        merged_bins = []
        i = 0
        while i < len(bins):
            s, e = bins[i]
            count = e - s
            if count < min_count:
                if merged_bins:
                    prev_s, prev_e = merged_bins[-1]
                    merged_bins[-1] = (prev_s, e)
                else:
                    if i + 1 < len(bins):
                        next_s, next_e = bins[i+1]
                        merged_bins.append((s, next_e))
                        i += 1  # Skip the next bin
                    else:
                        merged_bins.append((s, e))
            else:
                merged_bins.append((s, e))
            i += 1

        # It may be possible that after merging, an internal bin is still too small.
        changed = True
        while changed and len(merged_bins) > 1:
            changed = False
            new_bins = []
            i = 0
            while i < len(merged_bins):
                s, e = merged_bins[i]
                if (e - s) < min_count and i > 0:
                    prev_s, prev_e = new_bins[-1]
                    new_bins[-1] = (prev_s, e)
                    changed = True
                else:
                    new_bins.append((s, e))
                i += 1
            merged_bins = new_bins

        # Assign bin labels using the merged bins, computing central value and bin size.
        bin_labels = [None] * n
        for bin_idx, (s, e) in enumerate(merged_bins):
            count = e - s
            lower = q2_vals[s]
            upper = q2_vals[e-1]
            center = (lower + upper) / 2
            bin_size = upper - lower
            label = f"{category} bin {bin_idx+1}: {center:.3f} ± {bin_size:.3f} (n={count})"
            for j in range(s, e):
                bin_labels[j] = label

        return pd.Series(bin_labels, index=indices)

    # --- Create bins for each Q² category using the new algorithm ---
    low_bins = create_bins_for_category_maximize(g1f1_df, "Low Q2", min_count=5, gap_factor=2.0)
    mid_bins = create_bins_for_category_maximize(g1f1_df, "Mid Q2", min_count=5, gap_factor=2.0)
    high_bins = create_bins_for_category_maximize(g1f1_df, "High Q2", min_count=5, gap_factor=2.0)

    # Combine the bin labels and assign them as the new Q² labels column
    all_bins = pd.concat([low_bins, mid_bins, high_bins])
    g1f1_df['Q2_labels'] = all_bins
    
    # make dataframe of DIS values (W>2 && Q2>1)
    #dis_df = dis_df[dis_df['W']>2.0]
    dis_df = dis_df[dis_df['Q2']>1.0]

    # combine Mingyu data and g1f1_df
    temp_df = pd.DataFrame(
        {
            "Q2": mingyu_df["Q2"],
            "W": mingyu_df["W.cal"],
            "X": mingyu_df["x"],
            "G1F1": mingyu_df["g1F1_3He"],
            "G1F1.err": mingyu_df["g1f1.err"],
            "Label": ["Mingyu" for x in range(len(mingyu_df["Q2"]))],
        }
    )

    dis_df = pd.concat([temp_df, dis_df], ignore_index=True) # add Mingyu data
    
    # temp_df.head()
    
    dis_df.head(100)

    return g1f1_df, g2f1_df, a1_df, a2_df, dis_df
