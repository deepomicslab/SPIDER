import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import squidpy as sq

def smooth_pattern(idata, n_neighbors=10):
    import sklearn
    neigh = sklearn.neighbors.NearestNeighbors(n_neighbors=10)
    neigh.fit(idata.obs[['row', 'col']])
    neighbor_idx = neigh.kneighbors(idata.obs[['row', 'col']], return_distance=False)

    smoothed_pattern_score = []
    for i in neighbor_idx:
        smoothed_pattern_score.append(np.sum(idata.obsm['pattern_score'][i], axis=0))
    smoothed_pattern_score = np.array(smoothed_pattern_score)
    idata.obsm['smooth_pattern_score'] = smoothed_pattern_score

def paga(idata, label, n_neighbors=30, min_dist=1):
    import umap
    embedding = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,random_state=52).fit_transform(idata.obsm['smooth_pattern_score'])
    idata.obsm['X_umap'] = embedding
    sc.pp.neighbors(idata, n_neighbors=20, use_rep='X_umap')
    sc.tl.paga(idata, groups=label)
    

def paga_default(idata, label):
    import umap
    embedding = umap.UMAP(random_state=52, n_components=3).fit_transform(idata.obsm['smooth_pattern_score'])
    idata.obsm['X_umap'] = embedding
    sc.pp.neighbors(idata, use_rep='X_umap')
    sc.tl.paga(idata, groups=label)
    
def paga_spot(idata, adata, label, n_neighbors=30, min_dist=1):
    import umap
    embedding = umap.UMAP(random_state=42, min_dist=0.8, n_neighbors=20).fit_transform(idata.uns['cell_pattern'])
    adata.obsm['X_umap'] = embedding
    sc.pp.neighbors(adata, n_neighbors=20, use_rep='X_umap')   
    sc.tl.paga(adata, groups=label)
    
def pseudotime(idata, root_label, root_id=0):
    idata.uns['iroot'] = np.flatnonzero(idata.obs['label']  == root_label)[root_id]
    sc.tl.dpt(idata)
    
def projection(idata):
    import cellrank as cr
    import networkx
    networkx.from_scipy_sparse_matrix = networkx.from_scipy_sparse_array
    sq.gr.spatial_neighbors(idata, key_added='spatial')
    pt=cr.kernels.PseudotimeKernel(idata, time_key='dpt_pseudotime')
    pt.compute_transition_matrix()
    ck = cr.kernels.ConnectivityKernel(idata,conn_key='spatial_connectivities')
    ck.compute_transition_matrix()
    k=9*pt+1*ck
    k.compute_transition_matrix()
    k.compute_projection(basis="spatial")
    
