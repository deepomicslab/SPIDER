import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def relabel_interface(idata, clust_key):
    node_labels_text = idata.uns['cell_meta'][clust_key]
    idata.obs[f'A_label_{clust_key}'] = node_labels_text.loc[idata.obs['A']].astype(str).to_numpy()
    idata.obs[f'B_label_{clust_key}'] = node_labels_text.loc[idata.obs['B']].astype(str).to_numpy()
    node_labels = idata.uns['cell_meta'][clust_key].astype('category').cat.codes
    idata.obs[f'A_label_int_{clust_key}'] = node_labels.loc[idata.obs['A']].to_numpy()
    idata.obs[f'B_label_int_{clust_key}'] = node_labels.loc[idata.obs['B']].to_numpy()
    idata.obs['label_1'] = idata.obs[f'A_label_int_{clust_key}'].astype(str) + idata.obs[f'B_label_int_{clust_key}'].astype(str)
    idata.obs['label_2'] = idata.obs[f'B_label_int_{clust_key}'].astype(str) + idata.obs[f'A_label_int_{clust_key}'].astype(str)
    idata.obs[f'{clust_key}_int'] = idata.obs[['label_1', 'label_2']].astype(int).max(axis=1).astype(str).astype('category')
    label_1 = idata.obs[f'A_label_{clust_key}'].astype(str) + '_' + idata.obs[f'B_label_{clust_key}'].astype(str).to_numpy()
    label_2 = idata.obs[f'B_label_{clust_key}'].astype(str) + '_' + idata.obs[f'A_label_{clust_key}'].astype(str).to_numpy()
    pick = idata.obs[['label_1', 'label_2']].astype(int).idxmax(axis=1).to_numpy()
    text_label = [label_1[i] if x=='label_1' else label_2[i] for i,x in enumerate(pick)]
    idata.obs[f'label_{clust_key}'] = text_label
    idata.obs[f'label_{clust_key}'] = idata.obs[f'label_{clust_key}'].astype('category')
    print(f'Added key label_{clust_key} in idata.obs')
    
def scored_spot_interface(idata):       
    belonging = {}
    cells = idata.uns['cell_meta'].index
    for i in cells:
        belonging[i] = []
    for pair in idata.obs.reset_index()[['index','A', 'B']].to_numpy():
        belonging[pair[1]].append(pair[0])
        belonging[pair[2]].append(pair[0])
    score = pd.DataFrame(idata.obsm['pattern_score'], index=idata.obs_names)
    df = pd.concat([score.loc[belonging[c]].mean() for c in cells], axis=1).T     
    df.index = cells
    idata.uns['cell_pattern'] = df
    print(f'Added key cell_pattern in idata.uns')

def interaction_spot_interface(idata):       
    belonging = {}
    cells = idata.uns['cell_meta'].index
    for i in cells:
        belonging[i] = []
    for pair in idata.obs.reset_index()[['index','A', 'B']].to_numpy():
        belonging[pair[1]].append(pair[0])
        belonging[pair[2]].append(pair[0])
    score = idata.to_df()
    df = pd.concat([score.loc[belonging[c]].mean() for c in cells], axis=1).T     
    df.index = cells
    idata.uns['cell_score'] = df
    print(f'Added key cell_score in idata.uns')

def adata_moranI(adata, out_f, n_jobs=10):
    import scanpy as sc
    import squidpy as sq
    adata_copy = adata.copy()
    sc.pp.normalize_total(adata_copy, target_sum=1e4)
    sc.pp.log1p(adata_copy)
    sc.pp.highly_variable_genes(adata_copy, flavor='seurat_v3', n_top_genes=5000)
    genes = adata_copy[:, adata_copy.var.highly_variable].var_names.values[:5000]

    sq.gr.spatial_neighbors(adata, key_added='spatial')
    sq.gr.spatial_autocorr(
        adata,
        genes=genes,
        mode="moran",
        n_perms=1000,
        n_jobs=n_jobs,
    )
    adata.uns['moranI'].to_csv(f'{out_f}svg_moranI.csv')
    
def get_marker_df(idata, logfoldchanges_threhold=1):
    marker_df = pd.concat([pd.DataFrame(idata.uns['rank_genes_groups']['pvals_adj']).melt(),
           pd.DataFrame(idata.uns['rank_genes_groups']['logfoldchanges']).melt()['value'],
           pd.DataFrame(idata.uns['rank_genes_groups']['names']).melt()['value']], axis=1)
    marker_df.columns = ['cluster', 'pvals_adj', 'logfoldchanges', 'names']
    marker_df  = marker_df[(marker_df.pvals_adj < 0.05) & (marker_df.logfoldchanges > logfoldchanges_threhold)]
    print(marker_df.cluster.value_counts())
    return marker_df

def save_runningtime(idata, save=''):
    obj = {}
    keys = ['scanpy_time', 'nnSVG_time', 'SOMDE_time', 'SpatialDE_time', 'SPARKX_time', 'scGCO_time', 'moranI_time', 'gearyC_time']
    for key in keys:
        try:
            obj[key] = idata.uns[key]
        except:
            obj[key] = None
    obj['interface'] = idata.shape[0]
    obj['LRI'] = idata.shape[1]
    if save != '':
        import json
        with open(save, 'w') as f:
            json.dump(obj, f)
    return obj

def compare_var(idata, idata_old):
    val1, count1 = np.unique(idata_old.obs[['A', 'B']].to_numpy().flatten(), return_counts=True)
    val2, count2 = np.unique(idata.obs[['A', 'B']].to_numpy().flatten(), return_counts=True)
    df1 = pd.DataFrame({'val': val1, 'n_interface_v1': count1}).set_index('val')
    df2 = pd.DataFrame({'val': val2, 'n_interface_v2': count2}).set_index('val')
    n_interface = df2.join(df1).fillna(0)
    n_interface['ideal_n_interface'] = idata.uns['cell_meta'].loc[n_interface.index, 'ideal_n_interface']
    y_true = n_interface['ideal_n_interface'].to_numpy()
    y_pred_v1 = n_interface['n_interface_v1'].to_numpy()
    y_pred_v2 = n_interface['n_interface_v2'].to_numpy()
    mae_v1 = mean_absolute_error(y_true, y_pred_v1)
    mae_v2 = mean_absolute_error(y_true, y_pred_v2)
    mse_v1 = mean_squared_error(y_true, y_pred_v1)
    mse_v2 = mean_squared_error(y_true, y_pred_v2) 
    r2_v1 = r2_score(y_true, y_pred_v1)  
    r2_v2 = r2_score(y_true, y_pred_v2)
    print(f'v1: MAE={mae_v1}, MSE={mse_v1}, R2={r2_v1}')
    print(f'v2: MAE={mae_v2}, MSE={mse_v2}, R2={r2_v2}')
    return n_interface, mae_v1, mae_v2, mse_v1, mse_v2, r2_v1, r2_v2, y_true, y_pred_v1, y_pred_v2

def compare_local_var(idata, idata_old):
    from scipy.spatial import KDTree
    
    if len(idata) != len(idata_old):
        import anndata
        missing_cell = [x for x in idata.obs_names if x not in idata_old.obs_names ]
        idata_coexp_df = idata_old.to_df().T
        idata_coexp_df['missing_cell'] = 0
        idata_old = anndata.AnnData(idata_coexp_df.T)
    
    points = idata.obsm['spatial']
    tree = KDTree(points)
    arr = []
    for k in [5, 10, 20, 50]:
        _, indices = tree.query(points, k=k)
        for var_names in range(idata.shape[1]):
            values = idata.X[:, var_names]
            differences = np.abs(values[:, np.newaxis] - values[indices[:, 1:]])
            mean_differences = np.mean(differences, axis=1)
            mean = np.mean(mean_differences)
            arr.append([k, 'Optimal Transport', mean])

            values = idata_old.X[:, var_names]
            differences = np.abs(values[:, np.newaxis] - values[indices[:, 1:]])
            mean_differences = np.mean(differences, axis=1)
            mean = np.mean(mean_differences)
            arr.append([k, 'Co-expression', mean])
    return pd.DataFrame(arr, columns=['n_neighbor', 'formulation', 'diff'])
