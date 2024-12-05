import numpy as np
from scipy import optimize
import pandas as pd
import anndata
import os
from os.path import exists
import squidpy as sq
import warnings
warnings.filterwarnings('ignore')
import time
from importlib import resources
from scipy import sparse
from joblib import Parallel, delayed
from tqdm import tqdm


def tf_corr(idata, adata, is_human, out_f, threshold=0.3, n_jobs=20, overwrite=False, step=None, pathways=[], keep_top_score=None, smooth=False):
    """
    Compute transcription factor correlations with ligand-receptor interactions.

    This function analyzes the given interface data and evaluates the correlations
    between each interaction profile and their target genes, saving the results to 
    specified output files.

    Parameters
    ----------
    idata : AnnData
        Input AnnData object containing interface data.
        
    adata : AnnData
        Input AnnData object containing count data.
        
    is_human : bool
        Indicator for whether the data pertains to human genes.
        
    out_f : str
        Output directory path where results will be saved.
        
    threshold : float, optional
        Minimum correlation coefficient threshold for calling TF support (default is 0.3).
        
    n_jobs : int, optional
        Number of parallel jobs to run during processing (default is 20).
        
    overwrite : bool, optional
        If True, existing output files will be overwritten (default is False).
        
    step : float, optional
        The step size for adjusting the threshold during the iterative process (default is None).
        
    pathways : list, optional
        A list of pathways to analyze; if empty, pathways will be loaded based on the input data.
        
    keep_top_score : int, optional
        The number of top scoring interactions to retain (default is None).
        
    smooth : bool, optional
        If True, applies smoothing to the expression data (default is False).

    Returns
    -------
    None
        The function saves results directly to the specified output files and updates 
        the `idata` object with correlation and score data.

    Notes
    -----
    - The function checks if the output directory exists and creates it if it does not.
    - If smoothing is enabled, it applies a kernel smoothing method based on spatial data.
    - Results are saved in sparse matrix format and CSV files for easy access and further analysis.

    Examples
    --------
    >>> tf_corr(idata, adata, adata.uns['is_human'], 'output_directory/', threshold=0.3, n_jobs=30, overwrite=True, step=None, keep_top_score=5)
    """
    lr_df = idata.var[['ligand', 'receptor']]
    if len(pathways) == 0:
        pathways = subset_pathway_df(load_pathway_df(is_human), lr_df, adata)
    else:
        pathways = subset_pathway_df(pathways, lr_df, adata)
    print(pathways.head())
    unique_genes = np.unique(pathways[['src', 'dest']])
    value_to_index = {value: idx for idx, value in enumerate(unique_genes)}
    pathways['src_idx'] = pathways['src'].map(value_to_index)
    pathways['dest_idx'] = pathways['dest'].map(value_to_index)
    lr_df['receptor_idx'] = lr_df['receptor'].map(value_to_index)
    lr_df = lr_df.replace(np.nan, -1)
    lr_df['receptor_idx'] = lr_df['receptor_idx'].astype(int)
    hop1, sub_hop1, sub_hop2, sub_hop3, with_sv_tf_r, \
        sv_target_genes, sv_target_genes_index = get_ref_hops(adata, pathways, unique_genes, lr_df, value_to_index)
    idata.var['with_sv_tf'] = lr_df['with_sv_tf']
    print(f'Testing {idata.var["with_sv_tf"].sum()}/{len(idata.var)} LRIs ({int(100*round(idata.var["with_sv_tf"].sum()/len(idata.var), 2))}%) with SV target genes.')
    if (overwrite) | (not exists( f'{out_f}sparse_tf_corr.npz')):
        if smooth:
            neighborA = idata.obs.groupby('A')['B'].apply(list)
            neighborB = idata.obs.groupby('B')['A'].apply(list)
            neighbor = pd.concat([neighborA, neighborB], axis=0).reset_index(drop=False)
            neighbor.columns = ['A', 'B']
            neighbor = pd.Series(neighbor.groupby('A')['B'].sum())

            adata_exp = pd.DataFrame(idata.uns['tf_count'], columns=idata.uns['tf_header'], index=adata.obs_names)
            adata_exp_copy = adata_exp.copy()
            for cell in adata_exp.index:
                if cell in neighbor.index:
                    related_spots = neighbor.loc[cell]
                    adata_exp_copy.loc[cell] =  (adata_exp.loc[cell] + algebraic_mean(related_spots, adata_exp)) * 0.5
            adata_exp = adata_exp_copy.to_numpy()
        
            smth_eta = 1
            smth_nu = 1
            smth_kernel = 'exp'
            from scipy.spatial import distance_matrix
            kernel_D = distance_matrix(adata.obsm["spatial"], adata.obsm["spatial"])
            for i in range(adata_exp.shape[1]):
                nzind = np.where(adata_exp[:,i] > 0)[0]
                phi = kernel_function(kernel_D[nzind,:][:,nzind], smth_eta, smth_nu, smth_kernel, normalization='unit_col_sum')
                adata_exp[nzind,i] = np.matmul( phi, adata_exp[nzind,i].reshape(-1,1) )[:,0]
            adata_exp = pd.DataFrame(adata_exp, columns=idata.uns['tf_header'], index=adata.obs_names)
        else:
            adata_exp = pd.DataFrame(idata.uns['tf_count'], columns=idata.uns['tf_header'], index=adata.obs_names)
            
        
        pseudo_count = 1e-10
        adata_exp[adata_exp < pseudo_count] = pseudo_count
        gene_name_map = {value: key for key, value in value_to_index.items()}
        # adata_exp = pd.DataFrame(adata_exp, index=adata.obs_names, columns=adata.var_names)
        adata_exp = adata_exp / adata_exp.max(axis=0).values
        major_df = run_cells(adata_exp, unique_genes, sv_target_genes, with_sv_tf_r, hop1, 
                                    sub_hop1, sub_hop2, sub_hop3, 
                                    sv_target_genes_index, gene_name_map, n_jobs=n_jobs)
        results_corrs, results_tf_scores = run_receptors(major_df, idata, n_jobs=n_jobs, keep_top_score=keep_top_score)
        
        # save correlaitons result
        sparse.save_npz(f'{out_f}sparse_tf_corr.npz', sparse.csr_matrix(results_corrs.values))
        pd.Series(results_corrs.columns).to_csv(f'{out_f}tf_corr_columns.csv')
        pd.Series(results_corrs.index).to_csv(f'{out_f}tf_corr_index.csv')
        
        # save score result
        sparse.save_npz(f'{out_f}sparse_tf_scores.npz', sparse.csr_matrix(results_tf_scores.values))
        pd.Series(results_tf_scores.columns).to_csv(f'{out_f}tf_scores_columns.csv')
        pd.Series(results_tf_scores.index).to_csv(f'{out_f}tf_scores_index.csv')


    sparse_mat = sparse.csr_matrix(sparse.load_npz(f'{out_f}sparse_tf_corr.npz'))
    columns = pd.read_csv(f'{out_f}tf_corr_columns.csv', index_col=0).to_numpy().flatten()
    index = pd.read_csv(f'{out_f}tf_corr_index.csv', index_col=0).to_numpy().flatten()
    idata.uns['tf_corr'] = pd.DataFrame(sparse_mat.A, index=index, columns=columns)
    
    sparse_mat = sparse.csr_matrix(sparse.load_npz(f'{out_f}sparse_tf_scores.npz'))
    columns = pd.read_csv(f'{out_f}tf_scores_columns.csv', index_col=0).to_numpy().flatten()
    index = pd.read_csv(f'{out_f}tf_scores_index.csv', index_col=0).to_numpy().flatten()
    idata.obsm['tf_score'] = pd.DataFrame(sparse_mat.A, index=index, columns=columns)
    weight = count_tf(idata, threshold=threshold) 
    if step is not None:
        while weight > 0.5 or threshold >= 0.1:
            threshold = round(threshold - step, 2)
            weight = count_tf(idata, threshold=threshold)
        

def count_tf(idata, threshold=0.3):
    tf_count = (idata.uns['tf_corr'] > threshold).sum(axis=1).sort_values(ascending=False)
    print(f'{sum(tf_count > 0)}/{idata.var["with_sv_tf"].sum()} LRIs ({int(100*round(sum(tf_count > 0)/idata.var["with_sv_tf"].sum(), 2))}%) are predicted to be activated by downstream genes (threshold={threshold}).')
    idata.var['tf_support_count'] = -1
    idata.var.loc[tf_count.index, 'tf_support_count'] = tf_count
    print(f'Excluded {sum(tf_count == 0)}/{idata.shape[1]} LRIs ({int(100*round(sum(tf_count == 0)/idata.shape[1], 2))}%) without SV support (threshold={threshold}).')
    return 1-sum(tf_count > 0)/idata.var["with_sv_tf"].sum()
        
def algebraic_mean(related_samples, df, alpha=0.3):
    # alpha is the portion for max
    values = df.loc[related_samples]
    mean = values.mean() * (1-alpha) + values.max() * alpha
    return mean

def kernel_function(x, eta, nu, kernel, normalization=None):
    if kernel == "exp":
        phi = np.exp(-np.power(x/eta, nu))
    elif kernel == "lorentz":
        phi = 1 / ( 1 + np.power(x/eta, nu) )
    else:
        phi = x
    if normalization == "unit_row_sum":
        phi = (phi.T / np.sum(phi.T, axis=0)).T
    elif normalization == "unit_col_sum":
        phi = phi / np.sum(phi, axis=0)
    return phi

def load_pathway_df(is_human):
    from importlib import resources
    with resources.path("spider.lrdb", "pathways.tsv") as pw_fn:
        pw_list = pd.read_csv(pw_fn, sep='\t', index_col=0)
    if is_human:
        print('Using human pathway dataset.')
        pw_list = pw_list[pw_list.species=='Human']
    else:
        print('Using mouse pathway dataset.')
        pw_list = pw_list[pw_list.species=='Mouse']
    return pw_list

def subset_pathway_df(pathways, lr_df, adata):
    pathways = pathways[['src', 'dest']]
    pathways.index = pathways['src'] + '_' + pathways['dest']
    # remove pathways that are in the lrdb by index as we only want the downstream targets
    pathways = pathways[~pathways.index.isin(lr_df.index)]
    pathways = pathways[pathways['src'].isin(adata.var_names) | pathways['dest'].isin(adata.var_names)] # to tolerate missing points in pathwaay
    pathways = pathways.drop_duplicates()
    return pathways

def get_ref_hops(adata, pathways, unique_genes, lr_df, value_to_index):
    from scipy import sparse
    hop1 = sparse.csr_matrix((np.ones(len(pathways)), (pathways['src_idx'], pathways['dest_idx'])),
                                shape=(len(unique_genes), len(unique_genes)))
    hop2 = hop1.dot(hop1)
    hop3 = hop2.dot(hop1)
    r_index = lr_df['receptor_idx'].unique()
    r_index = np.array([x for x in r_index if x != -1])
    
    svg = adata.var[adata.var['label'] != -1].index.to_numpy()
    sv_target_genes  = np.intersect1d(svg, unique_genes)
    sv_target_genes_index = np.array([value_to_index[gene] for gene in sv_target_genes])
    print(f'Found {len(sv_target_genes)} SV pathway nodes.')

    # sub_hop: receptor x sv target genes
    sub_hop1 = hop1[r_index, :][:, sv_target_genes_index].A
    sub_hop2 = hop2[r_index, :][:, sv_target_genes_index].A
    sub_hop3 = hop3[r_index, :][:, sv_target_genes_index].A
    sum_path_all_htop = (sub_hop1 + sub_hop2 + sub_hop3).sum(axis=1)
    with_sv_tf_r = r_index[sum_path_all_htop != 0]
    print(f'Found {len(with_sv_tf_r)}/{len(sum_path_all_htop)} receptors ({int(100*round(len(with_sv_tf_r)/len(sum_path_all_htop), 2))}%) with SV target genes.')
    sub_hop1 = hop1[with_sv_tf_r, :][:, sv_target_genes_index].A
    sub_hop2 = hop2[with_sv_tf_r, :][:, sv_target_genes_index].A
    sub_hop3 = hop3[with_sv_tf_r, :][:, sv_target_genes_index].A

    lr_df['with_sv_tf'] = 0
    lr_df.loc[lr_df['receptor_idx'].isin(with_sv_tf_r), 'with_sv_tf'] = 1
    return hop1, sub_hop1, sub_hop2, sub_hop3, with_sv_tf_r, sv_target_genes, sv_target_genes_index

def process_cell(cell, unique_genes, norm_exp, sv_target_genes, with_sv_tf_r, hop1,
                 sub_hop1, sub_hop2, sub_hop3,
                  sv_target_genes_index, gene_name_map, pseudo_count=1e-10):
    # build full exp for cell
    exp = np.zeros(len(unique_genes))
    for i, gene in enumerate(unique_genes):
        if gene in norm_exp.columns:
            exp[i] = norm_exp.loc[cell, gene]
    exp[exp < pseudo_count] = pseudo_count # set pseudo count to exp

    weighted_hop1 = hop1.multiply(exp).tocsr()
    weighted_hop2 = (weighted_hop1.dot(hop1)).multiply(exp).tocsr()
    weighted_hop3 = (weighted_hop2.dot(hop1)).multiply(exp).tocsr()

    weighted_sub_hop1 = weighted_hop1[with_sv_tf_r, :][:, sv_target_genes_index].A / sub_hop1
    weighted_sub_hop1[~np.isfinite(weighted_sub_hop1)] = 0
    weighted_sub_hop2 = weighted_hop2[with_sv_tf_r, :][:, sv_target_genes_index].A / sub_hop2
    weighted_sub_hop2[~np.isfinite(weighted_sub_hop2)] = 0
    weighted_sub_hop2 = np.power(weighted_sub_hop2, 1/2)
    weighted_sub_hop3 = weighted_hop3[with_sv_tf_r, :][:, sv_target_genes_index].A / sub_hop3
    weighted_sub_hop3[~np.isfinite(weighted_sub_hop3)] = 0
    weighted_sub_hop3 = np.power(weighted_sub_hop3, 1/3)
    
    weighted_sum_path_all_htop = np.nansum([weighted_sub_hop1, weighted_sub_hop2, weighted_sub_hop3], axis=0)
    count_df = pd.DataFrame(weighted_sum_path_all_htop, columns=sv_target_genes)
    count_df['receptor'] = [gene_name_map[x] for x in with_sv_tf_r]
    count_df['cell'] = cell
    return count_df

def run_cells(norm_exp, unique_genes, sv_target_genes, with_sv_tf_r, hop1,
              sub_hop1, sub_hop2, sub_hop3,
               sv_target_genes_index, gene_name_map, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_cell)(cell, unique_genes, norm_exp, sv_target_genes, with_sv_tf_r, hop1,
                              sub_hop1, sub_hop2, sub_hop3,
                               sv_target_genes_index, gene_name_map)\
            for cell in tqdm(norm_exp.index, desc="Processing cells")
    )
    major_df = pd.concat(results)
    print('0227:1525')

    return major_df

def process_receptor(r, major_df, idata, keep_top_score=10):
    receptor_df = major_df[major_df.receptor == r].set_index('receptor').reset_index(drop=True).set_index('cell')
    receptor_df = receptor_df.loc[:, receptor_df.sum(axis=0) > 0]

    lrs = idata.var_names[idata.var_names.str.split('_').str[1] == r].to_numpy().tolist()
    corrs = []
    sv_tf_scores = []
    for lr in lrs:
        j = np.argwhere(idata.var_names == lr)[0][0]
        receivers = []
        for i in range(len(idata)):
            if idata.layers['direction'][i,j] == 1:
                receivers.append(idata.obs.iloc[i]['B'])
            else:
                receivers.append(idata.obs.iloc[i]['A'])
        sv_tf_score = receptor_df.loc[receivers].reset_index(drop=True)
        # drop the columns that are all zeros
        sv_tf_score = sv_tf_score[sv_tf_score.columns[sv_tf_score.sum()!=0]]
        sv_tf_score.index = idata.obs_names
        sv_tf_score[lr] = idata.to_df()[lr]
        if sv_tf_score[sv_tf_score.columns[:-1]].max().max() == 0:
            continue
        
        corr = sv_tf_score.corr(method='pearson')[lr][:-1] # the last row if the lr
        # only keep the positive correlations
        corr = corr[corr>0].sort_values(ascending=False)
        corrs.append(corr)
        
        sv_tf_score = sv_tf_score.drop(lr, axis=1)[corr.index[:keep_top_score]]
        sv_tf_score.columns = [f'{lr}-{c}' for c in sv_tf_score.columns]
        sv_tf_scores.append(sv_tf_score)
    corrs = pd.concat(corrs, axis=1)
    sv_tf_scores = pd.concat(sv_tf_scores, axis=1)
    return [corrs, sv_tf_scores]

def run_receptors(major_df, idata, n_jobs=-1, keep_top_score=10):
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_receptor)(r, major_df, idata, keep_top_score) for r in tqdm(major_df.receptor.unique(), desc="Processing receptors")
    )
    corrs = [result[0] for result in results]
    sv_tf_scores = [result[1] for result in results]
    results_corrs = pd.concat(corrs, axis=1).T.astype(np.float16).replace(np.nan, 0)
    results_tf_scores = pd.concat(sv_tf_scores, axis=1).replace(1e-10, 0)
    return results_corrs, results_tf_scores


# V1 code

def abstract(idata, n_neighbors=10, alpha=0.3):
    # Method: Building abstract interfaces with self-organizing map
    from somde import SomNode
    df = idata.to_df().T
    corinfo = idata.obs
    corinfo["total_count"]=df.sum(0)
    X=corinfo[['row','col']].values.astype(np.float32)
    som = SomNode(X,n_neighbors)
    ndf,ninfo = som.mtx(df, alpha=alpha)
    meta_idata = anndata.AnnData(ndf.T)
    meta_idata.obs[['row', 'col', 'total_count']] = ninfo.to_numpy(dtype=float)
    meta_idata.obsm['spatial'] = meta_idata.obs[['row', 'col']].to_numpy()
    idata = som_mapping(som, idata, df)
    return som, idata, meta_idata

def som_mapping(som, idata, df):
    bsmc = som.som.bmus
    soml = []
    for i in np.arange(bsmc.shape[0]):
        u,v = bsmc[i]
        soml.append(v*som.somn+u)
    idata.obs['som_node'] = -1
    ids = np.sort(np.unique(np.array(soml)))
    count = 0
    for i in ids:
        idata.obs.loc[df.loc[:,np.array(soml)==i].columns,'som_node'] = count
        count += 1
    return idata

def meta_pattern_to_idata(idata, meta_idata):
    idata.obsm['pattern_score'] = meta_idata.obsm['pattern_score'][idata.obs['som_node'].to_numpy()]  
    idata.var = meta_idata.var
    for i in np.array(['SOMDE', 'SpatialDE', 'SpatialDE2', 'SPARKX', 'nnSVG', 'scGCO', 'gearyC', 'moranI'])[np.isin(['SOMDE', 'SpatialDE', 'SpatialDE2', 'SPARKX', 'nnSVG', 'scGCO', 'gearyC', 'moranI'],list(meta_idata.uns.keys()))]:
        idata.uns[i] = meta_idata.uns[i]
        try:
            idata.uns[i+"_time"] = meta_idata.uns[i+"_time"]
        except:
            pass
    print(f'Added key pattern_score in idata.obsm and method results and running time in uns')   
    
def tf_pattern_to_idata(idata_raw, idata):
    idata_raw.uns['tf_corr'] = idata.uns['tf_corr']
    idata_raw.obsm['pattern_score'] = idata.obsm['pattern_score']
    idata_raw.var['is_svi'] = 0
    idata_raw.var.loc[idata.var_names, 'is_svi'] = idata.var['is_svi']
    idata_raw.var['label'] = -1
    idata_raw.var.loc[idata.var_names, 'label'] = idata.var['label']
    corr_cols = [f'pattern_correlation_{x}' for x in range(idata_raw.obsm['pattern_score'].shape[1])]
    idata_raw.var[corr_cols] = 0
    idata_raw.var.loc[idata_raw[:,idata_raw.var['is_svi']==1].var_names,corr_cols] = idata.var[corr_cols]
    for i in np.array(['SOMDE', 'SpatialDE', 'SpatialDE2', 'SPARKX', 'nnSVG', 'scGCO', 'gearyC', 'moranI'])[np.isin(['SOMDE', 'SpatialDE', 'SpatialDE2', 'SPARKX', 'nnSVG', 'scGCO', 'gearyC', 'moranI'],list(idata.uns.keys()))]:
        idata_raw.uns[i] = idata.uns[i]
        try:
            idata_raw.uns[i+"_time"] = idata.uns[i+"_time"]
        except:
            pass
        
    
def find_svi(idata, out_f, overwrite, R_path, som=None, n_jobs=10, skip_metric=False):
    # Method: Identifying spatially variable LR interactions
    # Gaussian models
    svi_nnSVG(idata,out_f,R_path,overwrite, n_jobs=n_jobs)
    svi_SOMDE(idata,out_f,overwrite, som=som, n_jobs=n_jobs)
    svi_SpatialDE2_omnibus(idata,out_f,overwrite, n_jobs=n_jobs)
    # svi_SpatialDE2(idata,out_f,overwrite)
    svi_SpatialDE(idata,out_f,overwrite)    
    # Non-parametric covariance test
    svi_SPARKX(idata,out_f,R_path,overwrite, n_jobs=n_jobs)
    # HMRF
    svi_scGCO(idata,out_f,overwrite, n_jobs=n_jobs)
    # baseline auto-correlation metrics
    if not skip_metric:
        svi_moran(idata,out_f,overwrite, n_jobs=n_jobs)
        svi_geary(idata,out_f,overwrite, n_jobs=n_jobs)

def svi_moran(idata, work_dir, overwrite=False, n_jobs=10):
    try:
        t0=time.time()
        n_perms=1000
        if (overwrite) | (not exists( f'{work_dir}moranI.csv')):
            sq.gr.spatial_neighbors(idata, key_added='spatial')
            sq.gr.spatial_autocorr(
                idata,
                mode="moran",
                n_perms=n_perms,
                n_jobs=n_jobs,
            )
            idata.uns['moranI_time'] = time.time()-t0
            idata.uns['moranI'].to_csv(f'{work_dir}moranI.csv')
        result = pd.read_csv(f'{work_dir}moranI.csv', index_col=0)
        idata.uns['moranI'] = result
        print(f'Added key moranI in idata.uns')
    except:
        pass

def svi_geary(idata, work_dir,overwrite=False, n_jobs=10):
    try:
        t0=time.time()
        n_perms=1000
        if (overwrite) | (not exists( f'{work_dir}gearyC.csv')):
            sq.gr.spatial_neighbors(idata, key_added='spatial')
            sq.gr.spatial_autocorr(
                idata,
                mode="geary",
                n_perms=n_perms,
                n_jobs=n_jobs,
            )
            idata.uns['gearyC_time'] = time.time()-t0
            idata.uns['gearyC'].to_csv(f'{work_dir}gearyC.csv')
        result = pd.read_csv(f'{work_dir}gearyC.csv', index_col=0)
        idata.uns['gearyC'] = result
        print(f'Added key gearyC in idata.uns')
    except:
        pass

def svi_nnSVG(idata, work_dir, R_path, overwrite=False, n_jobs=10):
    # overwrite = True
    try:
        count_f = f'{work_dir}idata_count.csv'
        meta_f = f'{work_dir}idata_meta.csv'
        if (overwrite) | ((not exists(count_f)) & (not exists(meta_f))):
            idata.to_df().to_csv(count_f)
            idata.obs[['row', 'col']].to_csv(meta_f)
        if (overwrite) | (not exists( f'{work_dir}nnSVG.csv')):
            t0=time.time()
            with resources.path("spider.R_script", "run_nnSVG.R") as pw_fn:
                os.system(str(f'/bin/bash -c "{R_path} -f {pw_fn} {count_f} {meta_f} {work_dir} {n_jobs}"'))
            idata.uns['nnSVG_time'] = time.time()-t0
        result = pd.read_csv(f'{work_dir}nnSVG.csv', index_col=0)
        idata.uns['nnSVG'] = result
        print(f'Added key nnSVG in idata.uns')
    except Exception as e:
        print(e)
    
def scGCO_sv(locs, data_norm, cellGraph, gmmDict, smooth_factor=5, unary_scale_factor=100, label_cost=10, algorithm='expansion'):
    from itertools import repeat
    from functools import reduce
    import operator
    import statsmodels.stats.multitest as multi
    import scGCO

    results = [scGCO.compute_single_fixed_sf(locs, data_norm, cellGraph, gmmDict, w=None, n=None, smooth_factor=smooth_factor, unary_scale_factor=unary_scale_factor, label_cost=label_cost, algorithm=algorithm)]
    
    nnn = [results[i][0] for i in np.arange(len(results))]
    nodes = reduce(operator.add, nnn)
    ppp = [results[i][1] for i in np.arange(len(results))]
    p_values=reduce(operator.add, ppp)
    ggg = [results[i][2] for i in np.arange(len(results))]
    genes = reduce(operator.add, ggg)
    fff = [results[i][3] for i in np.arange(len(results))]
    s_factors = reduce(operator.add, fff)
    lll = [results[i][4] for i in np.arange(len(results))]
    pred_labels = reduce(operator.add, lll)
    mmm = [results[i][5] for i in np.arange(len(results))]
    model_labels = reduce(operator.add, mmm)

    best_p_values=[min(i) for i in p_values]
    fdr = multi.multipletests(np.array(best_p_values), method='fdr_bh')[1]   
    
    labels_array = np.array(pred_labels).reshape(len(genes), pred_labels[0].shape[0])
    data_array = np.array((genes, p_values, fdr,s_factors, nodes, model_labels), dtype=object).T
    t_array = np.hstack((data_array, labels_array))
    c_labels = ['p_value', 'fdr',  'smooth_factor', 'nodes','model_labels']
    for i in np.arange(labels_array.shape[1]) + 1:
        temp_label = 'label_cell_' + str(i)
        c_labels.append(temp_label)
    result_df = pd.DataFrame(t_array[:,1:], index=t_array[:,0], columns=c_labels)
    return result_df

def scGCO_log1p(data):
    '''
    log transform normalized count data
    
    :param file: data (m, n); 
    :rtype: data (m, n);
    '''
    from scipy.sparse import issparse
    if not issparse(data):
        return np.log1p(data)
    else:
        return data.log1p()

def scGCO_normalize_count_cellranger(data,Log=True, norm=True):
    '''
    normalize count as in cellranger
    
    :param file: data: A dataframe of shape (m, n);
    :rtype: data shape (m, n);
    '''
    normalizing_factor = np.sum(data, axis = 1)/np.median(np.sum(data, axis = 1)) 
    # change to to .to_numpy() since some np/pd versions report "ValueError: Multi-dimensional indexing (e.g. `obj[:, None]`) is no longer supported. Convert to a numpy array before indexing instead" 
    data = pd.DataFrame(data.values/normalizing_factor.to_numpy()[:,np.newaxis], columns=data.columns, index=data.index)
    # data = pd.DataFrame(data.values/normalizing_factor[:,np.newaxis], columns=data.columns, index=data.index)
    if Log==True:
        data=scGCO_log1p(data)
    else:
        data=data

    return data

def svi_scGCO(idata, work_dir, overwrite=False, n_jobs=10):
    try:
        import scGCO
        if (overwrite) | (not exists( f'{work_dir}scGCO.csv')):
            t0=time.time()
            result_dfs = []
            for smooth_factor in [1, 5, 10, 20]:
                for unary_scale_factor in [50, 100]:
                    if (overwrite) | (not exists(f'{work_dir}scGCO_{smooth_factor}_{unary_scale_factor}.csv')):
                        data_norm = idata.to_df()
                        data_norm = scGCO_log1p(data_norm)
                        exp = data_norm.mean(axis=1)
                        pos_key=['row', 'col']
                        locs = idata.obs[pos_key].to_numpy()
                        cellGraph= scGCO.create_graph_with_weight(locs, exp)
                        gmmDict= scGCO.gmm_model(data_norm)
                        result_df= scGCO_sv(locs, data_norm, cellGraph, gmmDict, 
                            smooth_factor = smooth_factor, unary_scale_factor=unary_scale_factor)
                        result_df.to_csv(f'{work_dir}scGCO_{smooth_factor}_{unary_scale_factor}.csv')
                        result_dfs.append(result_df)
                    else:
                        result_df = pd.read_csv(f'{work_dir}scGCO_{smooth_factor}_{unary_scale_factor}.csv', index_col=0)
                        result_dfs.append(result_df)
            result_dfs = pd.concat(result_dfs)
            scGCO.write_result_to_csv(result_dfs, f'{work_dir}scGCO.csv')
            idata.uns['scGCO_time'] = time.time()-t0
        result = pd.read_csv(f'{work_dir}scGCO.csv').groupby('Unnamed: 0')['fdr'].min().reset_index()
        idata.uns['scGCO'] = result
        print(f'Added key scGCO in idata.uns')
    except:
        pass
    
def svi_SPARKX(idata, work_dir, R_path, overwrite=False, n_jobs=10):
    try:
        count_f = f'{work_dir}idata_count.csv'
        meta_f = f'{work_dir}idata_meta.csv'
        if (overwrite) | ((not exists(count_f)) & (not exists(meta_f))):
            idata.to_df().to_csv(count_f)
            idata.obs[['row', 'col']].to_csv(meta_f)
        if (overwrite) | (not exists( f'{work_dir}SPARKX.csv')):
            t0=time.time()
            with resources.path("spider.R_script", "run_SPARKX.R") as pw_fn:
                pw_fn = '/home/lishiying/data6/SPIDER-paper/spider_st/R_script/run_SPARKX.R'
                os.system(str(f'/bin/bash -c "{R_path} -f {pw_fn} {count_f} {meta_f} {work_dir} {n_jobs}"'))
            idata.uns['SPARKX_time'] = time.time()-t0
        result = pd.read_csv(f'{work_dir}SPARKX.csv', index_col=0)
        idata.uns['SPARKX'] = result
        print(f'Added key SPARKX in idata.uns')
    except:
        pass
    
def svi_SpatialDE2(idata, work_dir, overwrite=False, n_jobs=10):
    try:
        if (overwrite) | (not exists(f'{work_dir}SpatialDE.csv')):
            from spider import SpatialDE2
            t0=time.time()
            np.random.seed(20230617)
            svg_full, individual = SpatialDE2.test(idata, omnibus=False)
            svg_full = pd.concat([svg_full.set_index('gene'), individual.loc[individual.groupby('gene').lengthscale.idxmin()].set_index('gene')], axis=1)
            svg_full.to_csv(f'{work_dir}SpatialDE.csv')
            individual.to_csv(f'{work_dir}SpatialDE_individual.csv')
            idata.uns['SpatialDE_time'] = time.time()-t0
        result = pd.read_csv(f'{work_dir}SpatialDE.csv', index_col=0)
        idata.uns['SpatialDE'] = result
        print(f'Added key SpatialDE in idata.uns')
    except:
        pass 
    
def svi_SpatialDE2_omnibus(idata, work_dir, overwrite=False, n_jobs=10):
    try:
        if (overwrite) | (not exists(f'{work_dir}SpatialDE2_omnibus.csv')):
            from spider import SpatialDE2
            t0=time.time()
            np.random.seed(20230617)
            svg_full, _ = SpatialDE2.test(idata, omnibus=True)
            svg_full = svg_full.set_index('gene')
            svg_full.to_csv(f'{work_dir}SpatialDE2_omnibus.csv')
            idata.uns['SpatialDE2_time'] = time.time()-t0
        result = pd.read_csv(f'{work_dir}SpatialDE2_omnibus.csv', index_col=0)
        idata.uns['SpatialDE2'] = result
        print(f'Added key SpatialDE2 in idata.uns')
    except Exception as e:
        print(e)
        pass 
    
def svi_SpatialDE(idata, work_dir, overwrite=False, normalized=True):
    try:
        if (overwrite) | (not exists(f'{work_dir}SpatialDE.csv')):
            import NaiveDE
            import SpatialDE
            t0=time.time()
            counts = idata.to_df()
            sample_info = idata.obs[['row', 'col']]
            sample_info.columns = ['x', 'y']
            sample_info['total_counts'] = counts.sum(axis=1)
            if normalized:
                resid_expr = counts
            else:
                counts = counts[sample_info['total_counts'] > 1]
                sample_info = sample_info[sample_info['total_counts'] >1]
                norm_expr = NaiveDE.stabilize(counts.T).T
                resid_expr = NaiveDE.regress_out(sample_info, norm_expr.T, 'np.log(total_counts)').T
            X = sample_info[['x', 'y']]
            results = SpatialDE.run(X, resid_expr)
            idata.uns['SpatialDE_time'] = time.time()-t0
            results.to_csv(f'{work_dir}SpatialDE.csv')
        result = pd.read_csv(f'{work_dir}SpatialDE.csv', index_col=0)
        idata.uns['SpatialDE'] = result
        print(f'Added key SpatialDE in idata.uns')
    except Exception as e:
        print(e)
        pass 

def svi_SOMDE(idata, work_dir, overwrite=False, som=None, n_jobs=10):
    try:
        if (overwrite) | (not exists(f'{work_dir}SOMDE.csv')):
            t0=time.time()
            if som is None:
                from somde import SomNode
                df = idata.to_df().T
                corinfo = idata.obs
                corinfo["total_count"]=df.sum(0)
                X=corinfo[['row','col']].values.astype(np.float32)
                if len(df) < 1000:
                    som = SomNode(X,4)
                else:
                    som = SomNode(X,10)
                ndf,ninfo = som.mtx(df)
            # nres = som.norm()
            from scipy import optimize
            from somde import regress_out
            phi_hat, _ = optimize.curve_fit(lambda mu, phi: mu + phi * mu ** 2, som.ndf.mean(1), som.ndf.var(1))
            dfm = np.log(som.ndf + 1. / (2 * np.abs(phi_hat[0])))
            try:
                som.nres = regress_out(som.ninfo, dfm, 'np.log(total_count)').T
            except:
                som.nres = dfm.T
            result, SVnum =som.run()
            result.to_csv(f'{work_dir}SOMDE.csv')
            idata.uns['SOMDE_time'] = time.time()-t0
        result = pd.read_csv(f'{work_dir}SOMDE.csv', index_col=0)
        idata.uns['SOMDE'] = result
        print(f'Added key SOMDE in idata.uns')
    except:
        pass
    
def combine_SVI(idata, threshold, svi_number=10):
    """
    Combine SVIs based on both SV candidates and TF support.

    This function attempts to identify SVIs from the 
    given input data. It first uses strict filtering, then falls back to relaxed filtering 
    or SOMDE results if the detected number of SVIs is below a specified threshold.

    Parameters
    ----------
    idata : AnnData
        Input AnnData object containing interaction data.

    threshold : float
        Threshold value for filtering SVI candidates based on significance.

    svi_number : int, optional
        The mimimum number of SVIs to identify (default is 10). The function will 
        attempt to ensure that at least this many SVIs are returned.

    Returns
    -------
    svi_df : DataFrame
        DataFrame containing results of all LRIs.

    svi_df_strict : DataFrame
        DataFrame containing strictly filtered SVIs based on the provided threshold and TF support.

    Notes
    -----
    - The function checks the number of detected SVIs and applies different filtering 
      strategies if the count is less than `svi_number`.
    - If TF support counts are available, genes without TF support are excluded from 
      the strict SVI results.
    - Updates the `is_svi` column in `idata.var` to indicate which genes are identified as SVIs.

    Examples
    --------
    >>> svi_df, svi_df_strict = combine_SVI(idata, threshold=0.05, svi_number=10)
    """
    svi_df, svi_df_strict = combine_SVI_strict(idata,threshold=threshold)
    if len(svi_df_strict) <= svi_number:
        print(f'Detected SVI number is less than {svi_number}, falling back to relaxed filtering.')
        svi_df, svi_df_strict = combine_SVI_Fisher(idata,threshold=threshold)
    if len(svi_df_strict) <= svi_number:
        print(f'Detected SVI number is less than {svi_number}, falling back to use SOMDE result only.')
        svi_df, svi_df_strict  = combine_SVI_somde(idata,threshold=threshold)
    org_len = len(svi_df_strict)
    if 'tf_support_count' in idata.var.columns:
        svi_df_strict = svi_df_strict[idata.var['tf_support_count'].loc[svi_df_strict.index] != 0]
        print(f'Excluding {org_len-len(svi_df_strict)} genes without TF support')
        print(f'{len(svi_df_strict)}/{len(svi_df)} tf-supported SVIs identified.')
    else:
        print('TF corr not previously calculated, falling back to sv pattern only')
        print(f'{len(svi_df_strict)}/{len(svi_df)} SVIs identified.')
    idata.var['is_svi'] = 0
    idata.var.loc[svi_df_strict.index, 'is_svi'] = 1
    return svi_df, svi_df_strict

def combine_SVI_Fisher(idata, threshold=0.05):
    from scipy.stats import combine_pvalues
    methods = np.array(['SOMDE', 'SpatialDE', 'SpatialDE2', 'SPARKX', 'nnSVG', 'scGCO', 'gearyC', 'moranI'])[np.isin(['SOMDE', 'SpatialDE', 'SpatialDE2', 'SPARKX', 'nnSVG', 'scGCO', 'gearyC', 'moranI'],list(idata.uns.keys()))]
    print(f'Using the results from SVI identification methods: {methods}')
    df = []
    for i in methods:
        if i == 'SOMDE':
            df.append(idata.uns[i].set_index('g')[['qval']].rename(columns = {'qval': i}))
        elif i == 'SpatialDE2':
            df.append(idata.uns[i][['padj']].rename(columns = {'padj': i}))
        elif i == 'SpatialDE':
            df.append(idata.uns[i].set_index('g')[['qval']].rename(columns = {'qval': i}))
        elif i == 'SPARKX':
            df.append(idata.uns[i][['adjustedPval']].rename(columns = {'adjustedPval': i}))
        elif i == 'nnSVG':
            df.append(idata.uns[i][['padj']].rename(columns = {'padj': i}))
        elif i == 'scGCO':
            df.append(idata.uns[i].set_index(idata.uns['scGCO'].columns[0])[['fdr']].rename(columns = {'fdr': i}))
    df = pd.concat(df, axis=1).fillna(1)
    
    arr = [combine_pvalues(x, method='fisher')[1] for x in df.to_numpy()]
    df['adj_pvalue'] = arr
    df_sub = df[df['adj_pvalue']<threshold]
    print(f'{len(df_sub)}/{len(df)} SVIs identified (threshold={threshold}).')
    idata.uns['svi'] = df
    return df, df_sub

def combine_SVI_Stouffer(idata, threshold=0.05):
    from scipy.stats import combine_pvalues
    methods = np.array(['SOMDE', 'SpatialDE', 'SpatialDE2', 'SPARKX', 'nnSVG', 'scGCO', 'gearyC', 'moranI'])[np.isin(['SOMDE', 'SpatialDE', 'SpatialDE2', 'SPARKX', 'nnSVG', 'scGCO', 'gearyC', 'moranI'],list(idata.uns.keys()))]
    print(f'Using the results from SVI identification methods: {methods}')
    df = []
    for i in methods:
        if i == 'SOMDE':
            df.append(idata.uns[i].set_index('g')[['qval']].rename(columns = {'qval': i}))
        elif i == 'SpatialDE2':
            df.append(idata.uns[i][['padj']].rename(columns = {'padj': i}))
        elif i == 'SpatialDE':
            df.append(idata.uns[i].set_index('g')[['qval']].rename(columns = {'qval': i}))
        elif i == 'SPARKX':
            df.append(idata.uns[i][['adjustedPval']].rename(columns = {'adjustedPval': i}))
        elif i == 'nnSVG':
            df.append(idata.uns[i][['padj']].rename(columns = {'padj': i}))
        elif i == 'scGCO':
            df.append(idata.uns[i].set_index(idata.uns['scGCO'].columns[0])[['fdr']].rename(columns = {'fdr': i}))
    df = pd.concat(df, axis=1).fillna(1)
    
    arr = [combine_pvalues(x, method='stouffer')[1] for x in df.to_numpy()]
    df['adj_pvalue'] = arr
    df_sub = df[df['adj_pvalue']<threshold]
    print(f'{len(df_sub)}/{len(df)} SVIs identified (threshold={threshold}).')
    idata.uns['svi'] = df
    return df, df_sub

def combine_SVI_strict(idata, threshold=0.01):
    methods = np.array(['SOMDE', 'SpatialDE', 'SpatialDE2', 'SPARKX', 'nnSVG', 'scGCO', 'gearyC', 'moranI'])[np.isin(['SOMDE', 'SpatialDE', 'SpatialDE2', 'SPARKX', 'nnSVG', 'scGCO', 'gearyC', 'moranI'],list(idata.uns.keys()))]
    print(f'Using the results from SVI identification methods: {methods}')
    df = []
    for i in methods:
        if i == 'SOMDE':
            df.append(idata.uns[i].set_index('g')[['qval']].rename(columns = {'qval': i}))
        elif i == 'SpatialDE2':
            df.append(idata.uns[i][['padj']].rename(columns = {'padj': i}))
        elif i == 'SpatialDE':
            df.append(idata.uns[i].set_index('g')[['qval']].rename(columns = {'qval': i}))
        elif i == 'SPARKX':
            df.append(idata.uns[i][['adjustedPval']].rename(columns = {'adjustedPval': i}))
        # elif i == 'SPARKX':
        #     df.append(idata.uns[i].set_index('rn')[['fdr']].rename(columns = {'fdr': i}))
        elif i == 'nnSVG':
            df.append(idata.uns[i][['padj']].rename(columns = {'padj': i}))
        elif i == 'scGCO':
            df.append(idata.uns[i].set_index(idata.uns['scGCO'].columns[0])[['fdr']].rename(columns = {'fdr': i}))
    df = pd.concat(df, axis=1).fillna(1)
    df_sub = df[(df<threshold).all(axis=1)]
    print(f'{len(df_sub)}/{len(df)} SVIs identified (threshold={threshold}).')
    idata.uns['svi'] = df
    return df, df_sub

def combine_SVI_somde(idata, threshold=0.01):
    print(f'Using the SOMDE results')
    df = idata.uns['SOMDE'].set_index('g')[['qval']].rename(columns = {'qval': 'SOMDE'}).fillna(1)
    df_sub = df[(df<threshold).all(axis=1)]
    print(f'{len(df_sub)}/{len(df)} SVIs identified (threshold={threshold}).')
    idata.uns['svi'] = df
    return df, df_sub
    
def eva_SVI_notf(idata, svi_df_strict):
    import seaborn as sns
    from statannotations.Annotator import Annotator
    methods = np.array(['moranI', 'gearyC', 'SOMDE', 'nnSVG'])[np.isin(['SOMDE', 'nnSVG', 'gearyC', 'moranI'],list(idata.uns.keys()))]
    print(f'evaluating with {methods}')
    dfs = []
    metrics = []
    for i in methods:
        if i == 'gearyC':
            dfs.append(-idata.uns['gearyC'][['C']])
            metrics.append("Geary\nC (rev.)")
        elif i == 'moranI':
            dfs.append(idata.uns['moranI'][['I']]),
            metrics.append("Moran\nI")
        elif i == 'SOMDE':
            dfs.append(idata.uns['SOMDE'].set_index('g')['FSV']),
            metrics.append("FSV\n(SOMDE)") 
            dfs.append(idata.uns['SOMDE'].set_index('g')['LLR']),
            metrics.append("LR\n(SOMDE)") 
        elif i == 'nnSVG':
            dfs.append(idata.uns['nnSVG']['LR_stat']),
            metrics.append("LR\n(nnSVG)") 

    df = pd.concat(dfs, axis=1)
    df.columns=metrics

    normalized_df=(df-df.min())/(df.max()-df.min())
    normalized_df = normalized_df.fillna(0)
    normalized_df['Category'] = 'Excluded'
    normalized_df.loc[svi_df_strict.index, 'Category'] = 'SVI'
    normalized_df = normalized_df.melt(id_vars='Category', value_vars=metrics, var_name='Metric')
    normalized_df = normalized_df.sort_values('Category')
    if normalized_df['Category'].nunique()!=1:
        ax = sns.boxplot(data=normalized_df,x='Metric',y='value', hue='Category', palette={'SVI':'#80b1d3','Excluded': '#fb8072'}, width=0.8, hue_order=['SVI', 'Excluded'])
        pairs = []
        for i in metrics:
            pairs.append( ((i, 'SVI'), (i, 'Excluded')))
        annot = Annotator(ax, pairs, data=normalized_df, x='Metric',y='value', hue='Category', hue_order=['SVI', 'Excluded'])
        annot.configure(test='Mann-Whitney-gt',comparisons_correction="BH", correction_format="replace")
        annot.apply_test()
        annot.annotate()
    else:
        ax =sns.boxplot(data=normalized_df,x='Metric',y='value', hue='Category', palette={'SVI':'#80b1d3'}, width=0.8, hue_order=['SVI'])
        
    ax.legend(loc='upper center',ncol=2, bbox_to_anchor=(0.5, 1.1), frameon=False)

    ax.set_ylabel('')    
    ax.set_xlabel('')

def eva_SVI(idata, svi_df_strict):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from statannotations.Annotator import Annotator
    methods = np.array(['moranI', 'gearyC', 'SOMDE', 'nnSVG'])[np.isin(['moranI', 'gearyC', 'SOMDE', 'nnSVG'],list(idata.uns.keys()))]
    print(f'evaluating with {methods}')
    dfs = []
    metrics = []
    for i in methods:
        if i == 'gearyC':
            dfs.append(-idata.uns['gearyC'][['C']])
            metrics.append("Geary\nC (rev.)")
        elif i == 'moranI':
            dfs.append(idata.uns['moranI'][['I']]),
            metrics.append("Moran\nI")
        elif i == 'SOMDE':
            dfs.append(idata.uns['SOMDE'].set_index('g')['FSV']),
            metrics.append("FSV\n(SOMDE)") 
            dfs.append(idata.uns['SOMDE'].set_index('g')['LLR']),
            metrics.append("LR\n(SOMDE)") 
        elif i == 'nnSVG':
            dfs.append(idata.uns['nnSVG']['LR_stat']),
            metrics.append("LR\n(nnSVG)") 
            
    dfs.append(idata.uns['tf_corr'].max(axis=1))
    metrics.append('TF corr')

    df = pd.concat(dfs, axis=1)
    df.columns=metrics

    normalized_df=(df-df.min())/(df.max()-df.min())
    normalized_df['Category'] = 'Excluded'
    normalized_df.loc[svi_df_strict.index, 'Category'] = 'SVI'
    normalized_df = normalized_df.melt(id_vars='Category', value_vars=metrics, var_name='Metric').dropna()
    plt.figure(figsize=(6, 4))
    plt.rc('font', size=12)
    if normalized_df['Category'].nunique()!=1:
        ax = sns.boxplot(data=normalized_df,x='Metric',y='value', hue='Category', palette={'SVI':'#F7994A','Excluded': '#7F46C0'}, width=0.8, hue_order=['SVI', 'Excluded'])
        # ax = sns.boxplot(data=normalized_df,x='Metric',y='value', hue='Category', palette={'SVI':'#80b1d3','Excluded': '#fb8072'}, width=0.8, hue_order=['SVI', 'Excluded'])
        pairs = []
        for i in metrics:
            pairs.append( ((i, 'SVI'), (i, 'Excluded')))
        annot = Annotator(ax, pairs, data=normalized_df, x='Metric',y='value', hue='Category', hue_order=['SVI', 'Excluded'])
        annot.configure(test='Mann-Whitney-gt',comparisons_correction="BH", correction_format="replace")
        annot.apply_test()
        annot.annotate()
    else:
        ax =sns.boxplot(data=normalized_df,x='Metric',y='value', hue='Category', palette={'SVI':'#80b1d3'}, width=0.8, hue_order=['SVI'])
        
    ax.legend(loc='upper center',ncol=2, bbox_to_anchor=(0.5, 1.15), frameon=False)

    ax.set_ylabel('')    
    ax.set_xlabel('')
    return normalized_df
        
def eva_pattern(idata):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statannotations.Annotator import Annotator
    dfs = []
    pairs = []
    dummy_df = pd.get_dummies(idata.var['label'])
    dummy_df = pd.concat([idata.var[[f'pattern_correlation_{i}' for i in range(idata.obsm['pattern_score'].shape[1])]+['is_svi']], dummy_df], axis=1)
    dummy_df = dummy_df[dummy_df['is_svi'] == 1]
    for i in range(idata.obsm['pattern_score'].shape[1]):
        subdf = dummy_df[[i, f'pattern_correlation_{i}']]
        subdf.columns = ['membership', 'correlation']
        subdf['pattern'] = i
        dfs.append(subdf)
        pairs.append(((i, 'member'), (i, 'non-member')))
    maindf = pd.concat(dfs)
    maindf = maindf.sort_values('membership')
    maindf['membership'] = maindf['membership'].replace(0, 'non-member')
    maindf.loc[maindf['membership']!='non-member', 'membership'] = 'member'
    ax=sns.boxplot(data=maindf, x='pattern', y='correlation', hue='membership', hue_order=['member', 'non-member'],
                   palette={'member':'#F7994A','non-member': '#7F46C0'}, width=0.8)
    ax.legend(loc='upper center',ncol=2, bbox_to_anchor=(0.5, 1.15), frameon=False)
    annot = Annotator(ax, pairs, data=maindf, x='pattern',y='correlation', hue='membership')
    annot.configure(test='Mann-Whitney-ls',comparisons_correction="BH", correction_format="replace", fontsize=13)
    annot.apply_test()
    annot.annotate()
    plt.xlabel('SPIDER SVI patterns', fontsize=12)
    plt.title('SPIDER SVI pattern evaluation', y=1.1, fontsize=12)
    plt.ylabel('LRI-pattern correlations', fontsize=12)
    
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# SVI pattern generation with Gaussian process mixture model
def SVI_patterns(idata, svi_df_strict, iter=1000, pattern_prune_threshold=1e-6, predefined_pattern_number=-1):
    from spider import SpatialDE2
    allsignifgenes = svi_df_strict.index.to_numpy()

    if predefined_pattern_number == -1:
        param_obj = SpatialDE2.SpatialPatternParameters(maxiter=iter, pattern_prune_threshold=pattern_prune_threshold)
    else:
        param_obj = SpatialDE2.SpatialPatternParameters(maxiter=iter, pattern_prune_threshold=pattern_prune_threshold, nclasses=predefined_pattern_number)
    upper_patterns, _ = SpatialDE2.spatial_patterns(idata, normalized=True, genes=allsignifgenes, rng=np.random.default_rng(seed=45), params=param_obj, copy=False)
    if (predefined_pattern_number == -1) & ((upper_patterns.patterns.shape[1] > 20) | (upper_patterns.patterns.shape[1] < 5) | (len(np.unique(upper_patterns.patterns)) == 1)):
        param_obj = SpatialDE2.SpatialPatternParameters(maxiter=iter, pattern_prune_threshold=pattern_prune_threshold, nclasses=20)
        upper_patterns, _ = SpatialDE2.spatial_patterns(idata, normalized=True, genes=allsignifgenes, rng=np.random.default_rng(seed=45), params=param_obj, copy=False)
    if ((upper_patterns.patterns.shape[1] < 3) | (len(np.unique(upper_patterns.patterns)) == 1)) :
        param_obj = SpatialDE2.SpatialPatternParameters(maxiter=iter, pattern_prune_threshold=pattern_prune_threshold, nclasses=5)
        upper_patterns, _ = SpatialDE2.spatial_patterns(idata, normalized=True, genes=allsignifgenes, rng=np.random.default_rng(seed=45), params=param_obj, copy=False)
    # drop empty patterns
    labels, patterns, prob = upper_patterns.labels, upper_patterns.patterns, upper_patterns.pattern_probabilities
    uni_labels = np.sort(np.unique(labels))
    value_counts = np.array([np.sum(labels == i) for i in uni_labels])
    drop_labels = []
    for i in range(patterns.shape[1]):
        if len(np.unique(patterns[:, i])) == 1:
            uni_labels = np.delete(uni_labels, np.where(uni_labels == i))
            drop_labels.append(i)
    for i in uni_labels:
        if i >= patterns.shape[1]:
            drop_labels.append(i)
            uni_labels = np.delete(uni_labels, np.where(uni_labels == i))
    new_label_map = dict(zip(uni_labels, range(len(uni_labels))))
    for i in drop_labels:
        new_label_map[i] = -1
    print(new_label_map)
    patterns = patterns[:, uni_labels]
    prob = prob[:, uni_labels]
    new_labels = np.array([new_label_map[l] for l in labels])
    print(new_labels)
    upper_patterns = dotdict({
        'labels': new_labels,
        'patterns': patterns,
        'pattern_probabilities': prob
    })
    value_counts = np.array([np.sum(new_labels == i) for i in np.unique(new_labels)])
    print(value_counts)
    if ((upper_patterns.patterns.shape[1] < 3) | (len(np.unique(upper_patterns.patterns)) == 1)) :
        print(f'using controlled pattern')
        if predefined_pattern_number == -1:
            predefined_pattern_number = 10
        histology_results, patterns, prob = SVI_patterns_v1(idata, svi_df_strict, components=predefined_pattern_number)
        patterns = patterns.to_numpy()
        labels = histology_results['pattern'].to_numpy()
        uni_labels = np.sort(np.unique(labels))
        patterns = patterns[:, uni_labels]
        prob = prob[:, uni_labels]
        new_label_map = dict(zip(uni_labels, range(len(uni_labels))))
        new_labels = np.array([new_label_map[l] for l in labels])
        histology_results['pattern'] = new_labels
        upper_patterns = dotdict({
            'labels': histology_results['pattern'].to_numpy(),
            'patterns': patterns,
            'pattern_probabilities': prob
        })
    
    print(f'eventually found {upper_patterns.patterns.shape[1]} patterns')
    idata.var['label'] = -1
    if len(upper_patterns.labels) !=  len(allsignifgenes):
        allsignifgenes = svi_df_strict.index.to_numpy()
    print(upper_patterns.labels)
    idata.var.loc[allsignifgenes, 'label'] = upper_patterns.labels
    idata.var[[f'pattern_membership_{x}' for x in range(upper_patterns.pattern_probabilities.shape[1])]] = 0
    idata.var.loc[allsignifgenes, [f'pattern_membership_{x}' for x in range(upper_patterns.pattern_probabilities.shape[1])]] = upper_patterns.pattern_probabilities
    idata.obsm['pattern_score'] = upper_patterns.patterns
    idata.var[[f'pattern_correlation_{x}' for x in range(idata.obsm['pattern_score'].shape[1])]] = 0
    corr_df=pd.concat([idata.to_df(),pd.DataFrame(idata.obsm['pattern_score'],index=idata.obs_names)],axis=1).corr().loc[idata.var_names, range(idata.obsm['pattern_score'].shape[1])]
    idata.var[[f'pattern_correlation_{x}' for x in range(idata.obsm['pattern_score'].shape[1])]] = corr_df.to_numpy()
    
def SVI_patterns_v2(idata, svi_df_strict, iter=1000, pattern_prune_threshold=1e-8, predefined_pattern_number=-1):
    from spider import SpatialDE2
    allsignifgenes = svi_df_strict.index.to_numpy()
    if 'lengthscale' in idata.uns['SpatialDE'].columns:
        l=idata.uns['SpatialDE'].loc[allsignifgenes]['lengthscale'].to_list()
        if len(np.unique(l)) < 2:
            allsignifgenes = np.intersect1d(allsignifgenes, idata.uns['SOMDE']['g'].to_numpy())
            l=idata.uns['SOMDE'].set_index('g').loc[allsignifgenes]['l'].to_list()
    else:
        l=idata.uns['SOMDE'].set_index('g').loc[allsignifgenes]['l'].to_list()

    pattern_number = 1000
    print(predefined_pattern_number)
    if (len(np.unique(l)) <= 1) | (predefined_pattern_number != -1):
        pattern_number = -1
    for count in range(5):
        print(pattern_number, count, pattern_prune_threshold)
        if (pattern_number > 100) and (pattern_prune_threshold<1) and (count < 4):
            param_obj = SpatialDE2.SpatialPatternParameters(lengthscales=l,maxiter=iter, pattern_prune_threshold=pattern_prune_threshold)
            upper_patterns, _ = SpatialDE2.spatial_patterns(idata, genes=allsignifgenes, rng=np.random.default_rng(seed=45), params=param_obj, copy=False)
            pattern_number = upper_patterns.patterns.shape[1]
            pattern_prune_threshold = pattern_prune_threshold*100
            if pattern_number < 2:
                pattern_number = -1
        elif (pattern_number < 100) and (pattern_number > 2):
            break
        else:
            print(f'using controlled pattern')
            if predefined_pattern_number == -1:
                predefined_pattern_number = 5
        
            histology_results, patterns, prob = SVI_patterns_v1(idata, svi_df_strict, components=predefined_pattern_number)
            upper_patterns = dotdict({
                'labels': histology_results['pattern'].to_numpy(),
                'patterns': patterns.to_numpy(),
                'pattern_probabilities': prob
            })
            pattern_number = patterns.shape[1]
            break
    print(f'eventually found {pattern_number} patterns')
    idata.var['label'] = -1
    if len(upper_patterns.labels) !=  len(allsignifgenes):
        allsignifgenes = svi_df_strict.index.to_numpy()
    idata.var.loc[allsignifgenes, 'label'] = upper_patterns.labels
    idata.var[[f'pattern_membership_{x}' for x in range(upper_patterns.pattern_probabilities.shape[1])]] = 0
    idata.var.loc[allsignifgenes, [f'pattern_membership_{x}' for x in range(upper_patterns.pattern_probabilities.shape[1])]] = upper_patterns.pattern_probabilities
    idata.obsm['pattern_score'] = upper_patterns.patterns
    idata.var[[f'pattern_correlation_{x}' for x in range(idata.obsm['pattern_score'].shape[1])]] = 0
    corr_df=pd.concat([idata.to_df(),pd.DataFrame(idata.obsm['pattern_score'],index=idata.obs_names)],axis=1).corr().loc[idata.var_names, range(idata.obsm['pattern_score'].shape[1])]
    idata.var[[f'pattern_correlation_{x}' for x in range(idata.obsm['pattern_score'].shape[1])]] = corr_df.to_numpy()

    
def SVI_patterns_v1(idata, svi_df_strict, components=5):
    import NaiveDE
    import SpatialDE
    np.random.seed(20230617)
    
    df = idata.to_df().T.loc[svi_df_strict.index]
    corinfo = idata.obs
    corinfo["total_counts"]=df.sum(0)
    X=corinfo[['row','col']].values.astype(np.float32)
    # norm_expr = NaiveDE.stabilize(df).T
    from scipy import optimize
    phi_hat, _ = optimize.curve_fit(lambda mu, phi: mu + phi * mu ** 2, df.mean(1), df.var(1))
    dfm = np.log(df + 1. / (2 * np.abs(phi_hat[0])))
    nres = NaiveDE.regress_out(corinfo, dfm, 'np.log(total_count)').T
    # resid_expr = NaiveDE.regress_out(corinfo, norm_expr.T, 'np.log(total_counts)').T
    print('finished regression')
    results = SpatialDE.run(X,nres) # for generating lengthscale
    print('finished lengthscale')
    histology_results, patterns, prob = spatial_patterns(X, nres, results, C=components,l=results['l'].median()+0.5, verbosity=1)
    print('finished fitting')
    return histology_results, patterns, prob
    
def idata_pattern_to_spot(idata):
    belonging = {}
    cells = idata.uns['cell_meta'].index
    for i in cells:
        belonging[i] = []
    for pair in idata.obs.reset_index()[['index', 'A', 'B']].to_numpy():
        belonging[pair[1]].append(pair[0])
        belonging[pair[2]].append(pair[0])
    score = pd.DataFrame(idata.obsm['pattern_score'], index=idata.obs_names)
    df = pd.concat([score.loc[belonging[c]].mean() for c in cells], axis=1).T     
    df.index = cells
    idata.uns['cell_pattern'] = df
    print(f'Added key cell_pattern in idata.uns')   
    return df

def pattern_label_corr(data, pattern_df, label_key):
    from sklearn.feature_selection import mutual_info_classif
    label_df = pd.get_dummies(data.obs[label_key]).T
    mi=pd.DataFrame([mutual_info_classif(pattern_df, x) for x in label_df.to_numpy()], index=label_df.index, columns=pattern_df.columns)
    corr = pd.concat([pattern_df, label_df.T], axis=1).corr().loc[label_df.index,  pattern_df.columns]
    return mi, corr

# code created by SpatialDE
def spatial_patterns(X, exp_mat, DE_mll_results, C, l, **kwargs):
    ''' Group spatially variable genes into spatial patterns using
    automatic expression histology (AEH).
    X : Spatial coordinates
    exp_mat : Expression matrix, appropriately normalised.
    DE_mll_results : Results table from SpatialDE, after filtering
        for significance level.
    C : The number of spatial patterns
    **kwards are passed on to the function fit_patterns()
    Returns
    pattern_results : A DataFrame with pattern membership information
        for each gene
    patterns : The posterior mean underlying expression for genes in
        given spatial patterns.
    '''
    Y = exp_mat[DE_mll_results['g']].values.T

    # This is important, we only care about co-expression, not absolute levels.
    Y = (Y.T - Y.mean(1)).T
    Y = (Y.T / Y.std(1)).T

    _, m, r, _ = fit_patterns(X, Y, C, l, **kwargs)

    cres = pd.DataFrame({'g': DE_mll_results['g'],
                         'pattern': r.argmax(1),
                         'membership': r.max(1)})

    m = pd.DataFrame.from_records(m)
    m.index = exp_mat.index

    return cres, m, r


def make_elbojective(Y, r, m, X, K_0, s2e_0, pi=None):
    def elbojective(log_s2e):
        return -ELBO(Y, r, m, np.exp(log_s2e), K_0, K_0, s2e_0, pi)
    
    return elbojective

def SE_kernel(X, l):
    X = np.array(X)
    Xsq = np.sum(np.square(X), 1)
    R2 = -2. * np.dot(X, X.T) + (Xsq[:, None] + Xsq[None, :])
    R2 = np.clip(R2, 1e-12, np.inf)
    return np.exp((-R2 / (2 * l ** 2)).astype(float))

def ELBO(Y, r, m, s2e, K, K_0, s2e_0, pi=None):
    L = ln_P_YZms(Y, r, m, s2e, pi) + ln_P_Z(r, pi) + ln_P_mu(m, K) \
        - ln_Q_Z(r, r) - ln_Q_mu(K_0, r, s2e_0)
    
    return L

def factor(K):
    S, U = np.linalg.eigh(K)
    # .clip removes negative eigenvalues
    return U, np.clip(S, 1e-8, None)

def ln_Q_mu(K, Z, s2e):
    ''' Expecation of ln Q(mu)
    '''
    N = K.shape[0]
    C = Z.shape[1]
    G_k = Z.sum(0)
    
    ll = 0
    U, S = factor(K)
    for k in range(C):
        ll = ll - (1. / S + G_k[k] / s2e).sum()
        ll = ll + N * np.log(2 * np.pi)
        
    
    ll = -0.5 * ll
    
    return ll

def ln_Q_Z(Z, r):
    ''' Expectation of ln Q(Z)
    '''
    return np.sum(Z * np.log(r))

def ln_P_YZms(Y, Z, mu, s2e, pi=None):
    ''' Expecation of ln P(Y | Z, mu, s2e)
    '''
    G = Y.shape[0]
    N = Y.shape[1]
    C = Z.shape[1]
    if pi is None:
        pi = np.ones(C) / C
    
    log_rho = np.log(pi[None, :]) \
              - 0.5 * N * np.log(s2e) \
              - 0.5 * np.sum((mu.T[None, :, :] - Y[:, None, :]) ** 2, 2) / s2e \
              - 0.5 * N * np.log(2 * np.pi)
            
    return (Z * log_rho).sum()

def ln_P_Z(Z, pi=None):
    ''' Expectation of ln P(Z)
    '''
    C = Z.shape[1]
    if pi is None:
        pi = np.ones(C) / C
        
    return np.dot(Z, np.log(pi)).sum()

def ln_P_mu(mu, K):
    ''' Expectation of ln P(mu)
    '''
    N = K.shape[0]
    C = mu.shape[1]
    ll = 0
    for k in range(C):
        ll = ll + np.linalg.det(K)
        ll = ll + mu[:, k].dot(np.linalg.solve(K, mu[:, k]))
        ll = ll + N * np.log(2 * np.pi)
        
    ll = -0.5 * ll
    
    return ll

def fit_patterns(X, Y, C, l, s2e_0=1.0, verbosity=0, maxiter=100, printerval=1, opt_interval=1, delta_elbo_threshold=1e-2):
    ''' Fit spatial patterns using Automatic Expression Histology.
    X : Spatial coordinates
    Y : Gene expression values
    C : The number of patterns
    l : The charancteristic length scale of the clusters
    Returns
    final_elbo : The final ELBO value.
    m : The posterior mean underlying expression values for each cluster.
    r : The posterior pattern assignment probabilities for each gene and pattern.
    s2e : The estimated noise parameter of the model
    '''
    # Set up constants
    G = Y.shape[0]
    N = Y.shape[1]
    eps = 1e-8 * np.eye(N)
    
    s2e = s2e_0
    
    K = SE_kernel(X, l) + eps
    
    # Randomly initialize
    r = np.random.uniform(size=(G, C))
    r = r / r.sum(0)
    
    pi = r.sum(0) / G

    m = np.random.normal(size=(N, C))
    
    elbo_0 = ELBO(Y, r, m, s2e, K, K, s2e, pi)
    elbo_1 = elbo_0

    if verbosity > 0:
        print('iter {}, ELBO: {:0.2e}'.format(0, elbo_1))

    if verbosity > 1:
        print()

    for i in range(maxiter):
        if (i % opt_interval == (opt_interval - 1)):
            elbojective = make_elbojective(Y, r, m, X, K, s2e, pi)
            
            o = optimize.minimize_scalar(elbojective)
            s2e = np.exp(o.x)
            
            
        r = Q_Z_expectation(m, Y, s2e, N, C, G, pi)
        m = Q_mu_expectation(r, Y, K, s2e)
        
        pi = r.sum(0) / G

        elbo_0 = elbo_1
        elbo_1 = ELBO(Y, r, m, s2e, K, K, s2e, pi)
        delta_elbo = np.abs(elbo_1 - elbo_0)

        if verbosity > 0 and (i % printerval == 0):
            print('iter {}, ELBO: {:0.2e}, delta_ELBO: {:0.2e}'.format(i + 1, elbo_1, delta_elbo))
            
            if verbosity > 1:
                print('ln(l): {:0.2f}, ln(s2e): {:.2f}'.format(np.log(l), np.log(s2e)))
                
            if verbosity > 2:
                line1 = 'P(Y | Z, mu, s2e): {:0.2e}, P(Z): {:0.2e}, P(mu): {:0.2e}' \
                        .format(ln_P_YZms(Y, r, m, s2e, pi), ln_P_Z(r, pi), ln_P_mu(m, K))
                line2 = 'Q(Z): {:0.2e}, Q(mu): {:0.2e}'.format(ln_Q_Z(r, r), ln_Q_mu(K, r, s2e))
                print(line1 + '\n' + line2)
            
            if verbosity > 1:
                print()
            
        if delta_elbo < delta_elbo_threshold:
            if verbosity > 0:
                print('Converged on iter {}'.format(i + 1))

            break
            
    else:
        print('Warning! ELBO dit not converge after {} iters!'.format(i + 1))

    final_elbo = ELBO(Y, r, m, s2e, K, K, s2e, pi)
        
    return final_elbo, m, r, s2e

def Q_Z_expectation(mu, Y, s2e, N, C, G, pi=None):
    if pi is None:
        pi = np.ones(C) / C

    log_rho = np.log(pi[None, :]) \
              - 0.5 * N * np.log(s2e) \
              - 0.5 * np.sum((mu.T[None, :, :] - Y[:, None, :]) ** 2, 2) / s2e \
              - 0.5 * N * np.log(2 * np.pi)

    # Subtract max per row for numerical stability, and add offset from 0 for same reason.
    rho = np.exp(log_rho - log_rho.max(1)[:, None]) + 1e-12
    # Then evaluate softmax
    r = (rho.T / (rho.sum(1))).T
    
    return r

def Q_mu_k_expectation(Z_k, Y, K, s2e):
    y_k_tilde = np.dot(Z_k, Y) / s2e
    Sytk = np.dot(K, y_k_tilde)
    IpSDk = np.eye(K.shape[0]) + K * Z_k.sum() / s2e
    m_k = np.linalg.solve(IpSDk, Sytk)
    
    return m_k


def Q_mu_expectation(Z, Y, K, s2e):
    m = np.zeros((Y.shape[1], Z.shape[1]))

    y_k_tilde = np.dot(Z.T, Y).T / s2e

    for k in range(Z.shape[1]):
        m[:, k] = Q_mu_k_expectation(Z[:, k], Y, K, s2e)

    return m
    