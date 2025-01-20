#! /usr/bin/python

#
# Description:
# ================================================================
# Time-stamp: "2025-01-15 13:25:11 trottar"
# ================================================================
#
# Author:  Richard L. Trotta III <trottar.iii@gmail.com>
#
# Copyright (c) trottar
#
import pandas as pd

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

    # temp_df.head()

    dis_df = g1f1_df
    #dis_df = pd.concat([temp_df, g1f1_df], ignore_index=True) # add Mingyu data
    print(dis_df.head(100))


    # make dataframe of DIS values (W>2 && Q2>1)
    dis_df = dis_df[dis_df['W']>2.0]
    dis_df = dis_df[dis_df['Q2']>1.0]

    dis_df.head(100)

    return g1f1_df, g2f1_df, a1_df, a2_df, dis_df
