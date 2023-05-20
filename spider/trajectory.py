import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import squidpy as sq

def paga(idata, label):
    import umap, umap.plot
    embedding = umap.UMAP(n_neighbors=30, min_dist=1,random_state=52).fit_transform(idata.obsm['smooth_pattern_score'])
    idata.obsm['X_umap'] = embedding
    sc.pl.umap(idata,color='label')
    sc.pp.neighbors(idata, n_neighbors=20, use_rep='X_umap')
    sc.tl.paga(idata, groups=label)
    
def pseudotime(idata, root_label):
    idata.uns['iroot'] = np.flatnonzero(idata.obs['label']  == root_label)[0]
    sc.tl.dpt(idata)
    
def projection(idata):
    import cellrank as cr
    sq.gr.spatial_neighbors(idata, key_added='spatial')
    pt=cr.kernels.PseudotimeKernel(idata, time_key='dpt_pseudotime')
    pt.compute_transition_matrix()
    ck = cr.kernels.ConnectivityKernel(idata,conn_key='spatial_connectivities')
    ck.compute_transition_matrix()
    k=8*pt+2*ck
    k.compute_transition_matrix()
    k.plot_projection(basis="spatial")
    
