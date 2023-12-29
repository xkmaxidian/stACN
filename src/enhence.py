import logging

import scanpy as sc
import mnmstpy as mnmst

def enhence_data(AnnData):
    # enhance
    enhanced_adata, cell_spatial = mnmst.data_enhance(AnnData, k_nei=6)
    sc.pp.pca(enhanced_adata)
    return enhanced_adata,cell_spatial