import scanpy as sc
import pandas as pd
import numpy as np

def supervised_spot_clust(idata, adata, label, portion=0.1, n_cluster=None, n_neighbors=100, min_dist=1):
    import umap
    idata.uns['cell_meta'][label+'_int'] = idata.uns['cell_meta'][label].astype('category').cat.codes
    masked_target = idata.uns['cell_meta'][label+'_int']
    np.random.seed(52)
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
        # print(len(adata.obs['leiden'].unique()), res)
        if (len(adata.obs['leiden'].unique()) == k) | (res < 0) | (res > 10) | (repeat > 1000):
            flag = 0
        elif len(adata.obs['leiden'].unique()) < k:
            res_min = res
            res = (res+res_max)/2
        else:
            res_max = res
            res = (res+res_min)/2
    adata.obs['leiden_supervised'] = adata.obs['leiden']
    
            
def unsupervised_spot_clust(idata, adata, n_cluster=None):
    import umap

    embedding = umap.UMAP(n_neighbors=100, min_dist=1,random_state=52).fit_transform(idata.uns['cell_pattern'])
    adata.obsm['X_umap'] = embedding
    sc.pp.neighbors(adata, n_neighbors=20, use_rep='X_umap')
    sc.tl.draw_graph(adata)
    flag = 1
    res_max = 10
    res_min = 0.0001
    res = 0.05
    k = n_cluster
    while (flag):
        sc.tl.leiden(adata, resolution=res)
        # print(len(adata.obs['leiden'].unique()), res)
        if (len(adata.obs['leiden'].unique()) == k) | (res < 0) | (res > 10):
            flag = 0
        elif len(adata.obs['leiden'].unique()) < k:
            res_min = res
            res = (res+res_max)/2
        else:
            res_max = res
            res = (res+res_min)/2
    adata.obs['leiden_unsupervised'] = adata.obs['leiden']

def interface_clustering(idata, n_cluster=15):
    flag = 1
    res_max = 10
    res_min = 0.0001
    res = 0.05
    k = n_cluster
    sc.pp.neighbors(idata, n_neighbors=52, use_rep='X_umap')
    while (flag):
        sc.tl.leiden(idata, resolution=res,random_state=52)
        # print(len(idata.obs['leiden'].unique()), res)
        if (len(idata.obs['leiden'].unique()) == k) | (res < 0) | (res > 10):
            flag = 0
        elif len(idata.obs['leiden'].unique()) < k:
            res_min = res
            res = (res+res_max)/2
        else:
            res_max = res
            res = (res+res_min)/2