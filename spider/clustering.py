import scanpy as sc
import numpy as np
import umap

def supervised_spot_clust(idata, adata, label, portion=0.1, n_cluster=None, n_neighbors=100, min_dist=1, random_seed=52):
    masked_target = idata.uns['cell_meta'][label].astype('category').cat.codes.to_numpy().copy()
    np.random.seed(random_seed)
    masked_target[np.random.choice(len(idata.uns['cell_meta'][label]), size=int(len(idata.uns['cell_meta'][label]) * portion), replace=False)] = -1
    embedding = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,random_state=52).fit_transform(idata.uns['cell_pattern'], y=masked_target)
    adata.obsm['X_umap'] = embedding
    sc.pp.neighbors(adata, n_neighbors=20, use_rep='X_umap')
    sc.tl.draw_graph(adata)
    flag = 1
    res_max = 10
    res_min = 0.0001
    res = 0.05
    if not n_cluster:
        k = len(idata.uns['cell_meta'][label].unique())
    else:
        k = n_cluster
    repeat = 0
    while (flag):
        sc.tl.leiden(adata, resolution=res)
        repeat+=1
        if (len(adata.obs['leiden'].unique()) == k) | (res < 0) | (res > 10) | (repeat > 1000):
            flag = 0
        elif len(adata.obs['leiden'].unique()) < k:
            res_min = res
            res = (res+res_max)/2
        else:
            res_max = res
            res = (res+res_min)/2
    adata.obs['leiden_supervised'] = adata.obs['leiden']
    
            
def unsupervised_spot_clust(idata, adata, n_cluster=None, n_neighbors=100, min_dist=1):
    embedding = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,random_state=52).fit_transform(idata.uns['cell_pattern'])
    adata.obsm['X_umap'] = embedding
    sc.pp.neighbors(adata, n_neighbors=20, use_rep='X_umap')
    sc.tl.draw_graph(adata)
    flag = 1
    res_max = 10
    res_min = 0.0001
    res = 0.05
    k = n_cluster
    repeat = 0
    while (flag):
        sc.tl.leiden(adata, resolution=res)
        repeat+=1
        if (len(adata.obs['leiden'].unique()) == k) | (res < 0) | (res > 10) | (repeat > 1000):
            flag = 0
        elif len(adata.obs['leiden'].unique()) < k:
            res_min = res
            res = (res+res_max)/2
        else:
            res_max = res
            res = (res+res_min)/2
    adata.obs['leiden_unsupervised'] = adata.obs['leiden']
