import pandas as pd
import numpy as np
import scanpy as sc
import networkx as nx
import magic
import scprep
from pathlib import Path
from sklearn.metrics import pairwise_distances
from scipy.spatial import Delaunay
from sklearn.decomposition import PCA,NMF
import anndata 
from importlib import resources


# screen
def cci_spatalk(adata, work_dir, cluster_key, is_human, out_f, R_path):
    import os
    count_f = f'{work_dir}/adata_count.csv'
    meta_f = f'{work_dir}/adata_meta.csv'
    df = adata.to_df()
    df.index = "C"+df.index
    df.to_csv(count_f)
    meta = adata.obs[cluster_key].reset_index()
    meta[['x', 'y']] = adata.obsm['spatial']
    meta.columns = ['cell','celltype', 'x', 'y']
    meta.cell = "C"+meta.cell
    meta = meta[['cell','x', 'y', 'celltype']]
    if not pd.api.types.is_string_dtype(meta.celltype.dtype):
        meta.celltype = "T"+meta.celltype.astype('str')
    meta.celltype = meta.celltype.str.replace(' ', '_')
    meta.celltype = meta.celltype.str.replace('-', '_')
    meta.to_csv(meta_f)
    species = 'Human' if is_human else 'Mouse'
    with resources.path("spider.R_script", "run_spatalk.R") as pw_fn:
        os.system(str(f'/bin/bash -c "{R_path} -f {pw_fn} {count_f} {meta_f} {species} {out_f}"'))
    
# imputation
def impute_MAGIC(adata):
    magic_op = magic.MAGIC(n_jobs=5, random_state=0)
    inp = adata.to_df()
    inp = scprep.normalize.library_size_normalize(inp)
    inp = scprep.transform.sqrt(inp)
    outp = magic_op.fit_transform(inp)
    adata.X = outp

# idata
def idata_construct(score, pairs_meta, lr_df, lr_raw, adata):
    idata = anndata.AnnData(score)
    idata.obs_names = pairs_meta.index
    idata.var_names = lr_df.index
    idata.uns['lr_meta'] = lr_raw
    idata.obs = pairs_meta
    unique_cells = np.unique(idata.obs[['A', 'B']].to_numpy().flatten())
    cell_meta = adata.obs.loc[unique_cells]
    idata.uns['cell_meta'] = cell_meta
    # quality check
    sc.pp.calculate_qc_metrics(idata, inplace=True, percent_top=None)
    sc.pp.filter_genes(idata, min_cells=5)
    sc.pp.filter_cells(idata, min_genes=1)
    sc.pp.normalize_total(idata, target_sum=1e4)
    idata.obsm['spatial'] = idata.obs[['row', 'col']].to_numpy()
    print(f'Construct idata with {idata.shape[0]} interfaces and {idata.shape[1]} LR pairs.')
    return idata

def subset_lr(adata, no_spatalk, work_dir, cluster_key, is_human, overwrite, R_path):
    from os.path import exists
    if not no_spatalk:
        out_f = f'{work_dir}/spatalk'
        if overwrite | (not exists(f'{out_f}_lrpair.csv')):
            cci_spatalk(adata, work_dir, cluster_key, is_human, out_f, R_path)
        else:
            print(f'{out_f}_lrpair.csv already exists, skipping spatalk.')
    try:
        print('using spatalk result')
        lr_raw = pd.read_csv(f'{out_f}_lrpair.csv', index_col=0).sort_values('score')
        lr_raw = lr_raw.drop_duplicates(subset=['ligand', 'receptor'], keep="last")
    except:
        print('no spatalk result, using all lrpairs')
        lr_raw = load_lr_df(is_human).drop_duplicates(subset=['ligand', 'receptor'], keep="last")
        lr_raw['score'] = 1
    return lr_raw


def score(adata, lr_df, pairs):
    exp_ref = adata.to_df()
    exp_ref = exp_ref.loc[:,~exp_ref.columns.duplicated()]
    l = lr_df['ligand'].to_numpy().flatten()
    r = lr_df['receptor'].to_numpy().flatten()
    sub_exp = exp_ref[np.concatenate((l, r))].to_numpy()
    sub_exp_rev = exp_ref[np.concatenate((r, l))].to_numpy()
    edge_exp_both = np.multiply(sub_exp[pairs[0]], sub_exp_rev[pairs[1]])
    # equation 2 in the manuscript
    print('scoring')
    print('using sqrt+max')
    score = np.sqrt(np.maximum(edge_exp_both[:, :int(len(l))], edge_exp_both[:, int(len(l)):]))
    return score

def find_interfaces(adata, coord_type, n_neighs, cluster_key):
        pairs = find_pairs(adata, coord_type=coord_type, n_neighs=n_neighs)
        pairs_meta = meta(adata, cluster_key, pairs)
        return pairs, pairs_meta

def subset_adata(adata, lr_df, imputation):
    if imputation:
        print('Running imputation with MAGIC')
        impute_MAGIC(adata)
    
    genes = adata.var_names.tolist()
    lr_df = lr_df[lr_df['ligand'].isin(genes) & lr_df['receptor'].isin(genes)]
    lr_df.index = lr_df['ligand'] + "_" + lr_df['receptor']
    l = lr_df['ligand'].to_numpy().flatten()
    r = lr_df['receptor'].to_numpy().flatten()
    unique_lr = np.unique(np.concatenate((l, r)))
    adata = adata[:, adata.var_names.isin(unique_lr)]
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.filter_cells(adata, min_genes=1)
    sc.pp.normalize_total(adata, target_sum=1e4)
    genes = adata.var_names.tolist()
    lr_df = lr_df[lr_df['ligand'].isin(genes) & lr_df['receptor'].isin(genes)]
    return lr_df, adata


def find_pairs(adata, coord_type='generic', n_neighs=6):
    from squidpy.gr import spatial_neighbors
    from scipy.sparse import triu
    if coord_type == 'grid':
        spatial_neighbors(adata, coord_type=coord_type, n_neighs=n_neighs)
    else:
        spatial_neighbors(adata, coord_type=coord_type, delaunay=True, n_neighs=n_neighs)
    return np.transpose(triu(adata.obsp['spatial_connectivities']).nonzero()).T

def meta(adata, cluster_key, pairs):
    # get label
    pairs_meta = pd.DataFrame()
    pairs_meta['A'] = adata.obs_names[pairs[0]]
    pairs_meta['B'] = adata.obs_names[pairs[1]]
    pairs_meta[['A_row', 'A_col']] = adata.obsm['spatial'][pairs[0]]
    pairs_meta[['B_row', 'B_col']] =  adata.obsm['spatial'][pairs[1]]

    if cluster_key != '': 
        node_labels_text = adata.obs[cluster_key].to_numpy()
        pairs_meta['A_label'] = node_labels_text[pairs[0]].astype(str)
        pairs_meta['B_label'] = node_labels_text[pairs[1]].astype(str)
        node_labels = adata.obs[cluster_key].astype('category').cat.codes.to_numpy() + 1
        pairs_meta['A_label_int'] = node_labels[pairs[0]]
        pairs_meta['B_label_int'] = node_labels[pairs[1]]
        pairs_meta['label_1'] = pairs_meta["A_label_int"].astype(str) + pairs_meta["B_label_int"].astype(str)
        pairs_meta['label_2'] = pairs_meta["B_label_int"].astype(str) + pairs_meta["A_label_int"].astype(str)
        pairs_meta['label_int'] = pairs_meta[['label_1', 'label_2']].astype(int).max(axis=1).astype(str).astype('category')
        label_1 = pairs_meta['A_label'].astype(str) + '_' + pairs_meta['B_label'].astype(str).to_numpy()
        label_2 = pairs_meta['B_label'].astype(str) + '_' + pairs_meta['A_label'].astype(str).to_numpy()
        pick = pairs_meta[['label_1', 'label_2']].astype(int).idxmax(axis=1).to_numpy()
        text_label = [label_1[i] if x=='label_1' else label_2[i] for i,x in enumerate(pick)]
        pairs_meta['label'] = text_label
        pairs_meta['label'] = pairs_meta['label'].astype('category')

    pairs_meta.index = pairs_meta['A'] + "_" + pairs_meta['B']

    # get position  
    A_pos = pairs_meta[['A_row', 'A_col']].to_numpy(dtype=float)
    B_pos = pairs_meta[['B_row', 'B_col']].to_numpy(dtype=float)
    avg_pair_pos = (A_pos + B_pos) / 2
    pairs_meta[['row', 'col']] = avg_pair_pos
    pairs_meta['dist'] = np.linalg.norm(A_pos-B_pos, axis=1)
    return pairs_meta


def load_lr_df(is_human):
    from importlib import resources
    with resources.path("spider.lrdb", "lrpairs.tsv") as pw_fn:
        lr_list = pd.read_csv(pw_fn, sep='\t', index_col=0)
    if is_human:
        print('Using human LR pair dataset.')
        lr_list = lr_list[lr_list.species=='Human']
    else:
        print('Using mouse LR pair dataset.')
        lr_list = lr_list[lr_list.species=='Mouse']
    return lr_list



