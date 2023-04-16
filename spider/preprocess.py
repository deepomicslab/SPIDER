import pandas as pd
import numpy as np
import scanpy as sc
import networkx as nx
import magic
import scprep
from pathlib import Path
from sklearn.metrics import pairwise_distances
from scipy.spatial import Delaunay
PACKAGEDIR = Path(__file__).parent.absolute()
from sklearn.decomposition import PCA,NMF
import anndata 



# detect cci
def cci_spatalk(adata, work_dir, cluster_key, is_human, out_f):
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
    os.system(str(f'/bin/bash -c "source /etc/profile;module load GCC/11.2.0 OpenMPI/4.1.1 R/4.2.0 Anaconda3/2022.05 R-bundle-Bioconductor/3.15-R-4.2.0;R -f ../src/run_spatalk.R {count_f} {meta_f} {species} {out_f}"'))
    
# imputation
def impute_MAGIC(adata):
    magic_op = magic.MAGIC(n_jobs=5)
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
    idata.obsm['spatial'] = idata.obs[['row', 'col']].to_numpy()
    print(f'Construct idata with {idata.shape[0]} interactions and {idata.shape[1]} LR pairs.')
    return idata

def score(adata, lr_df, pairs):
    exp_ref = adata.to_df()
    exp_ref = exp_ref.loc[:,~exp_ref.columns.duplicated()]
    l = lr_df['ligand'].to_numpy().flatten()
    r = lr_df['receptor'].to_numpy().flatten()
    sub_exp = exp_ref[np.concatenate((l, r))].to_numpy()
    sub_exp_rev = exp_ref[np.concatenate((r, l))].to_numpy()
    edge_exp_both = np.multiply(sub_exp[pairs[0]], sub_exp_rev[pairs[1]])
    score = np.sqrt(np.maximum(edge_exp_both[:, :int(len(l))], edge_exp_both[:, int(len(l)):]))
    return score

def subset(adata, lr_df):
    genes = adata.var_names.tolist()
    lr_df = lr_df[lr_df['ligand'].isin(genes) & lr_df['receptor'].isin(genes)]
    lr_df.index = lr_df['ligand'] + "_" + lr_df['receptor']
    l = lr_df['ligand'].to_numpy().flatten()
    r = lr_df['receptor'].to_numpy().flatten()
    unique_lr = np.unique(np.concatenate((l, r)))
    adata = adata[:, adata.var_names.isin(unique_lr)]
    sc.pp.filter_genes(adata, min_cells=1)
    sc.pp.filter_cells(adata, min_genes=1)
    sc.pp.normalize_total(adata, target_sum=1e4)
    genes = adata.var_names.tolist()
    lr_df = lr_df[lr_df['ligand'].isin(genes) & lr_df['receptor'].isin(genes)]
    return lr_df, adata

def find_pairs(adata, coord_type='grid', n_neighs=6):
    from squidpy.gr import spatial_neighbors
    from scipy.sparse import triu
    if coord_type == 'grid':
        spatial_neighbors(adata, coord_type=coord_type, n_neighs=n_neighs)
    else:
        spatial_neighbors(adata, coord_type=coord_type, delaunay=True, n_neighs=n_neighs)
    return np.transpose(triu(adata.obsp['spatial_connectivities']).nonzero()).T

# def meta(adata, position_keys, cluster_key, pairs):
def meta(adata, cluster_key, pairs):
    # get label
    pairs_meta = pd.DataFrame()
    pairs_meta['A'] = adata.obs_names[pairs[0]]
    pairs_meta['B'] = adata.obs_names[pairs[1]]
    # pairs_meta[['A_row', 'A_col']] = adata.obs[position_keys].iloc[pairs[0]].to_numpy()
    # pairs_meta[['B_row', 'B_col']] = adata.obs[position_keys].iloc[pairs[1]].to_numpy()
    pairs_meta[['A_row', 'A_col']] = adata.obsm['spatial'][pairs[0]]
    pairs_meta[['B_row', 'B_col']] =  adata.obsm['spatial'][pairs[1]]

    if cluster_key != '': 
        node_labels_text = adata.obs[cluster_key].to_numpy()
        pairs_meta['A_label'] = node_labels_text[pairs[0]].astype(str)
        pairs_meta['B_label'] = node_labels_text[pairs[1]].astype(str)
        node_labels = adata.obs[cluster_key].astype('category').cat.codes.to_numpy()
        pairs_meta['A_label_int'] = node_labels[pairs[0]]
        pairs_meta['B_label_int'] = node_labels[pairs[1]]
        pairs_meta['label_1'] = pairs_meta["A_label_int"].astype(str) + pairs_meta["B_label_int"].astype(str)
        pairs_meta['label_2'] = pairs_meta["B_label_int"].astype(str) + pairs_meta["A_label_int"].astype(str)
        # pairs_meta.to_csv('/home/lishiying/data6/01-interaction/check.csv')
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


# deplicated

def eigen_lr(idata, adata, lr_df, k, pairs, method='NMF'):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # adata.obs["total_counts"] = np.ravel(X.sum(axis=1))
    # sc.pp.normalize_total(to_reduce, target_sum=1e4)
    # print
    # sc.pp.log1p(to_reduce)
    # print(to_reduce)
    # sc.pp.regress_out(to_reduce, ['total_counts'])
    exp_l = adata[:, lr_df['ligand'].to_numpy().flatten()].to_df().to_numpy()
    exp_r = adata[:, lr_df['receptor'].to_numpy().flatten()].to_df().to_numpy()
    if method == 'NMF':
        model = NMF(n_components=k, init='random', random_state=0)
        W = model.fit_transform(np.vstack([exp_l, exp_r]))
    elif method == 'PCA':
        model = PCA(n_components=k, random_state=0, svd_solver='randomized') 
        W = model.fit_transform(np.vstack([exp_l, exp_r]))
    # adata.obsm[f'{method}_eigen_l'] = W[:len(to_reduce)]
    # adata.obsm[f'{method}_eigen_r'] = W[len(to_reduce):]
    idata.obsm[f'AL_eigen_{method}'] = W[:len(adata)][pairs[0]]
    idata.obsm[f'AR_eigen_{method}'] = W[len(adata):][pairs[0]]
    idata.obsm[f'BL_eigen_{method}'] = W[:len(adata)][pairs[1]]
    idata.obsm[f'BR_eigen_{method}'] = W[len(adata):][pairs[1]]


def load_lr_df(is_human):
    from importlib import resources
    with resources.path("spider.lrdb", "lrpairs.tsv") as pw_fn:
        lr_list = pd.read_csv(pw_fn, sep='\t', index_col=0)
    # lr_list = pd.read_csv(f'{PACKAGEDIR}/lr_pair_data/lrpairs.tsv', sep='\t', index_col=0)
    # lr_list['weight'] = 1
    # lr_list = lr_list[lr_list.species=='Human']
    
    if is_human:
        print('Using human LR pair dataset.')
        lr_list = lr_list[lr_list.species=='Human']
    else:
        print('Using mouse LR pair dataset.')
        lr_list = lr_list[lr_list.species=='Mouse']
    return lr_list


    # if is_human:
    #     print('Using human LR pair dataset.')
    #     lr = pd.read_csv(f'{PACKAGEDIR}/lr_pair_data/human(org_clean).txt', sep='\t', header=None)
    # else:
    #     print('Using mouse LR pair dataset.')
    #     lr = pd.read_csv(f'{PACKAGEDIR}/lr_pair_data/mouse.txt', sep='\t', header=None)

    lr_list = lr_list[['ligand','receptor', 'weight']]
    lr_list.columns = [0, 1, 2]
    # print(lr_list)
    return lr_list

def subset_old(adata, lr_df):
    # first subset
    genes = adata.var_names.tolist()

    lr_df = lr_df[lr_df['ligand'].isin(genes) & lr_df['receptor'].isin(genes)]
    # print(f'{len(lr_list)} out of {len(lr)} LR pairs are applicable.')

    # lr to df
    # lr_df = pd.DataFrame(lr_list)
    # lr_df.columns = ['ligands', 'receptors', 'weight']
    lr_df.index = lr_df['ligand'] + "_" + lr_df['receptor']

    # subset adata
    l = lr_df['ligand'].to_numpy().flatten()
    r = lr_df['receptor'].to_numpy().flatten()
    unique_lr = np.unique(np.concatenate((l, r)))

    # subset adata
    adata = adata[:, adata.var_names.isin(unique_lr)]
    sc.pp.filter_genes(adata, min_cells=1)
    sc.pp.filter_cells(adata, min_genes=2)
    # sc.pp.filter_genes(adata, min_cells=20)


    # processing adata
    # sc.pp.calculate_qc_metrics(adata, inplace=True)
    sc.pp.normalize_total(adata, target_sum=1e4)
    # sc.pp.log1p(adata)
    # sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    # adata = adata[:, adata.var.highly_variable]
    # sc.pp.`regress_out`(adata, ['total_counts'])

    # second subset
    genes = adata.var_names.tolist()
    lr_df = lr_df[lr_df['ligand'].isin(genes) & lr_df['receptor'].isin(genes)]
    # if len(lr_df) < 10:
    #     raise Exception(f'Only {len(lr_df)} high-quality LR pairs out of {len(lr)} LR pairs are kept - Please check the quality of your data.')
    # else:
    #     print(f'{len(lr_df)} high-quality LR pairs out of {len(lr)} LR pairs are kept.')

    # second subset adata
    # l = lr_df['ligands'].to_numpy().flatten()
    # r = lr_df['receptors'].to_numpy().flatten()
    # unique_lr = np.unique(np.concatenate((l, r)))
    # adata = adata[:, unique_lr]
    # sc.pp.filter_genes(adata, min_cells=2)
    # sc.pp.filter_cells(adata, min_genes=1)
    return lr_df, adata

def nor_lr(lr_df):
    G = nx.Graph()
    edgeList = lr_df.to_numpy()
    for i in range(len(edgeList)):
        G.add_edge(edgeList[i][0], edgeList[i][1], weight=edgeList[i][2])
    A = nx.adjacency_matrix(G).A
    D = np.diag(1/np.sqrt(np.array(G.degree)[:, 1].astype(int)))
    nor_A = D @ A @ D 
    lr_df['weight'] = nor_A[np.nonzero(np.tril(nor_A, -1))]
    print(f'Weights of LR pairs are normalized.')

def impute_SMURF(adata):
    timer = TicToc()
    timer.tic()
    n_features = 10 if adata.shape[1] <= 100 else (20 if adata.shape[1] <= 500 else 50)
    operator = smurf.SMURF(n_features=n_features, estimate_only=True)
    adata.X = operator.smurf_impute(adata.to_df().T).T
    timer.toc('SMURF mputation finished in')



def impute_SAVER(adata):
    adata.to_df().to_csv(f'/home/lishiying/data3/mouse/{data[0]}/{data[1].split(".h5ad")[0]}_count.csv')
    
def find_pairs_v1(adata, position_keys, grid, n_neighs=6):
    dist = pairwise_distances(adata.obs[position_keys])
    if grid:
        if dist_offset == 0:
            dist = dist * (dist <= np.sort(dist, axis=1)[:,[20]]).astype(int)
        else:
            dist[dist > dist_offset] = 0
    else:
        print('non grid coordinate')
        tri = Delaunay(adata.obs[position_keys].to_numpy())
        indptr, indices = tri.vertex_neighbor_vertices
        adj = np.zeros((adata.shape[0], adata.shape[0]))
        for k in range(adata.shape[0]):
            neis = indices[indptr[k]:indptr[k+1]]
            for i in neis:
                adj[k][i] = 1
        adj += adj.T
        if dist_offset == 0:
            dist = dist * (dist <= np.quantile(dist, 0.5)).astype(int)
        else:
            dist[dist > np.quantile(dist, dist_offset)] = 0
        dist = dist * adj
    dist = np.tril(dist, -1)

    # weak_cells = np.where(~dist.any(axis=1))[0]

    # if len(weak_cells) != 0:
    #     adata_copy = adata_copy[~np.isin(np.arange(len(dist)), weak_cells)]
    #     dist = dist[~np.isin(np.arange(len(dist)), weak_cells)].T[~np.isin(np.arange(len(dist)), weak_cells)]
    #     print(f'dropped {len(weak_cells)} cells with no interaction with other cells.')
    
    return np.transpose(dist.nonzero()).T

# def normalize(adata):
#     timer = TicToc()
#     timer.tic()
#     librarySize = self.exp.sum(axis=0)
#     meanLibrarySize = librarySize.mean()
#     normalizedDataFrame = self.exp / (librarySize / meanLibrarySize)
#     self.exp = np.log(normalizedDataFrame+1)
#     timer.toc('Normalization finished in')

def count(adata, lr_df, edge_metric, pairs):
    exp_ref = adata.to_df()
    exp_ref = exp_ref.loc[:,~exp_ref.columns.duplicated()]
    l = lr_df['ligand'].to_numpy().flatten()
    r = lr_df['receptor'].to_numpy().flatten()

    # actual count
    AL = exp_ref.iloc[pairs[0]][l].to_numpy()
    AR = exp_ref.iloc[pairs[0]][r].to_numpy()
    BL = exp_ref.iloc[pairs[1]][l].to_numpy()
    BR = exp_ref.iloc[pairs[1]][r].to_numpy()
    # simplified expression
    sub_exp = exp_ref[np.concatenate((l, r))].to_numpy()
    sub_exp_rev = exp_ref[np.concatenate((r, l))].to_numpy()
    edge_exp_both = np.multiply(sub_exp[pairs[0]], sub_exp_rev[pairs[1]])
    if edge_metric == 'add':
        count = edge_exp_both[:, :int(len(l))] + edge_exp_both[:, int(len(l)):]
    elif edge_metric == 'max':
        count = np.maximum(edge_exp_both[:, :int(len(l))], edge_exp_both[:, int(len(l)):])
    direction = np.argmax((edge_exp_both[:, :int(len(l))], edge_exp_both[:, int(len(l)):]), 0)
    return AL, AR, BL, BR, direction, count


