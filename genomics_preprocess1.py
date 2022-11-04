import os
import pandas as pd
import numpy as np
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

dfrna = pd.read_table("rna_data from directory")
dfmethl= pd.read_table("dna_methylation data from directory")
dfmir = pd.read_table("mirna data from directory")


#removing features present in less than 50% from samples

def genePrep_hori_st(dfmirna):
    rw, cl = dfmirna.shape
    dfbio = dfmirna.iloc[:, 1: ]
    rw1, cl1 = dfbio.shape
    df_empty1 =  []
    df_empty2 =  []
    for i in range(rw1):
        rwdt = dfbio.iloc[i]
        rwdt1 = dfmirna.iloc[i]
        cnt = 0
        for j in range(cl1):
            if rwdt[j] == 0:
                cnt = cnt +1
        if cnt/cl1 < 0.5 or cnt == 0:
            df_empty1.append(rwdt1)
        else:
            df_empty2.append(rwdt1)
               
    return(pd.DataFrame(df_empty1))

dfrna_ftr = genePrep_hori_st(dfrna)
dfmir_ftr = genePrep_hori_st(dfmir)
dfmethl_ftr = genePrep_hori_st(dfmethl)
