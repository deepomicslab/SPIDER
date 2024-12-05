import pandas as pd
import numpy as np
import scanpy as sc
import magic
import scprep
import anndata 
from importlib import resources
from .ot import cot_combine_sparse
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull
import itertools
from sklearn.mixture import GaussianMixture
    
# imputation
def impute_MAGIC(adata):
    magic_op = magic.MAGIC(n_jobs=5, random_state=0)
    inp = adata.to_df()
    inp = scprep.normalize.library_size_normalize(inp)
    inp = scprep.transform.sqrt(inp)
    outp = magic_op.fit_transform(inp)
    adata.X = outp

# idata
def idata_construct(score, direction, pairs_meta, lr_df, lr_raw, pathway_df, adata, drop=True, normalize_total=False):
    """
    Constructs an AnnData object for interface profiles from input data.

    This function takes scores and metadata to create an AnnData object, 
    setting various attributes and performing quality checks on the data.

    Parameters
    ----------
    score : np.ndarray
        A 2D array of scores representing the interaction strengths between ligand-receptor pairs.
    direction : np.ndarray
        A 2D array indicating the direction of interactions for the respective scores.
    pairs_meta : pd.DataFrame
        A DataFrame containing metadata for ligand-receptor pairs, which should include relevant identifiers.
    lr_df : pd.DataFrame
        A DataFrame containing ligand-receptor information, with indices corresponding to ligand-receptor pairs.
    lr_raw : pd.DataFrame
        Raw metadata for ligand-receptor pairs, used for further analysis and storage in the output AnnData object.
    pathway_df : pd.DataFrame
        A DataFrame containing pathway information that describes interactions between transcription factors and genes.
    adata : AnnData
        The original AnnData object containing the input data for the analysis.
    drop : bool, optional
        If True, drops cells that do not have any expressed genes (default is True).
    normalize_total : bool, optional
        If True, normalizes the total counts of interactions in the AnnData object (default is False).

    Returns
    -------
    idata : AnnData
        The constructed AnnData object containing the following:
        - Scores and directions as layers.
        - Metadata and quality metrics for interfaces.
        - Spatial coordinates if available.

    Notes
    -----
    - The function performs quality checks on the data, filtering genes and cells based on specified criteria.
    - It normalizes total interaction strengths per interface if the `normalize_total` flag is set to True.
    - The function constructs the output AnnData object (`idata`) with appropriate metadata for downstream analysis.

    Examples
    --------
    >>> idata = idata_construct(score_array, direction_array, pairs_meta_df, lr_df, lr_raw_df, pathway_df, adata)
    """
    idata = anndata.AnnData(score)
    idata.layers['direction'] = direction
    idata.obs_names = pairs_meta.index
    idata.var_names = lr_df.index
    idata.var = lr_df
    idata.uns['lr_meta'] = lr_raw
    idata.uns['pathway_meta'] = pathway_df
    idata.obs = pairs_meta
    unique_cells = np.unique(idata.obs[['A', 'B']].to_numpy().flatten())
    cell_meta = adata.obs.loc[unique_cells]
    idata.uns['cell_meta'] = cell_meta
    idata.uns['tf_count'] = adata.to_df()[np.unique(pathway_df[['src', 'dest']])].to_numpy()
    idata.uns['tf_header'] = np.unique(pathway_df[['src', 'dest']])
    # quality check
    sc.pp.calculate_qc_metrics(idata, inplace=True, percent_top=None)
    sc.pp.filter_genes(idata, min_cells=5)
    if drop:
        sc.pp.filter_cells(idata, min_genes=1)
    if normalize_total:
        sc.pp.normalize_total(adata)
        print('Normalizing total counts per cell to the median of total counts for cells before normalization')
    print(f'Construct idata with {idata.shape[0]} interfaces and {idata.shape[1]} LR pairs.')
    return idata

def subset_lr(is_human):
    lr_raw = load_lr_df(is_human).drop_duplicates(subset=['ligand', 'receptor'], keep="last")
    lr_raw['score'] = 1
    return lr_raw

def build_neighbors(row, edges):
    source_node = row.A
    target_node = row.B
    return pd.Series({'A': [n for n in edges.loc[source_node]['neighbor'] if n != target_node],
                      'B': [n for n in edges.loc[target_node]['neighbor'] if n != source_node]})
    
def get_interface_neighbors(adata, interface_meta):
    neighborA = interface_meta.groupby('A')['B'].apply(list)
    neighborB = interface_meta.groupby('B')['A'].apply(list)
    neighbor = pd.concat([neighborA, neighborB], axis=0).reset_index(drop=False)
    neighbor.columns = ['A', 'B']
    neighbor = pd.Series(neighbor.groupby('A')['B'].sum())
    df = pd.DataFrame(index=adata.obs_names, columns=['neighbor'])
    df['neighbor'] = [[]] * len(df)
    df.loc[neighbor.index, 'neighbor'] = neighbor.to_numpy()
    # Function to get neighbors for the source node of an edge
    interface_neighbor = interface_meta.apply(build_neighbors, args=(df,), axis=1)
    return interface_neighbor

def algebraic_mean_v1(row, df, is_source, alpha=0.3):
    # alpha is the portion for max
    if is_source:
        related_samples = row.A
    else:
        related_samples = row.B
    values = df.loc[related_samples]
    mean = values.mean() * (1-alpha) + values.max() * alpha
    return mean

def algebraic_mean(related_samples, df, alpha=0.3):
    # alpha is the portion for max
    values = df.loc[related_samples]
    mean = values.mean() * (1-alpha) + values.max() * alpha
    return mean

def score(adata, lr_df, pairs, interface_meta):
    interface_neighbor = get_interface_neighbors(adata, interface_meta)
    exp_ref = adata.to_df()
    exp_ref = exp_ref.loc[:,~exp_ref.columns.duplicated()]
    l = lr_df['ligand'].to_numpy().flatten()
    r = lr_df['receptor'].to_numpy().flatten()
    sub_exp = exp_ref[np.concatenate((l, r))]
    sub_exp_rev = exp_ref[np.concatenate((r, l))]
    # Compute algebraic mean for each sample
    neighbor_exp_A = interface_neighbor.apply(algebraic_mean_v1, args=(sub_exp, True), axis=1)
    neighbor_exp_B = interface_neighbor.apply(algebraic_mean_v1, args=(sub_exp_rev, False), axis=1)
    mask_A = neighbor_exp_A.isna().any(axis=1)
    mask_B = neighbor_exp_B.isna().any(axis=1)
    neighbor_exp_A[mask_A] = sub_exp.loc[interface_meta.A[mask_A]].to_numpy()
    neighbor_exp_B[mask_B] = sub_exp_rev.loc[interface_meta.B[mask_B]].to_numpy()
    neighbor_exp_A = neighbor_exp_A.to_numpy()
    neighbor_exp_B = neighbor_exp_B.to_numpy()
    sub_exp = sub_exp.to_numpy()
    sub_exp_rev = sub_exp_rev.to_numpy()
    # multiply lr exp
    edge_exp_both = np.multiply(sub_exp[pairs[0]] + neighbor_exp_A, sub_exp_rev[pairs[1]] + neighbor_exp_B)/4
    print('scoring')
    print('using neighbor+sqrt+max')
    score = np.sqrt(np.maximum(edge_exp_both[:, :int(len(l))], edge_exp_both[:, int(len(l)):]))
    direction = np.argmax((edge_exp_both[:, :int(len(l))], edge_exp_both[:, int(len(l)):]), 0)
    return score, direction

def score_ot(adata, lr_df, interface_meta, interface_cell_pair, weight=(0.25, 0.25, 0.25, 0.25), itermax = 1000):
    """
    Score interactions and directions between ligands and receptors based on spatial data.

    This function evaluates the interactions between ligands and receptors in the provided 
    AnnData object, using a COT scoring mechanism that incorporates spatial information and 
    co-expression data.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object containing expression data and spatial coordinates.
        
    lr_df : DataFrame
        DataFrame containing ligand-receptor pairs to be analyzed.
        
    interface_meta : DataFrame
        Metadata associated with the interface analysis (not directly used in scoring).
        
    interface_cell_pair : DataFrame
        A DataFrame containing pairs of cell indices for which interactions are to be scored.
        
    weight : tuple of float, optional
        Weights for the scoring components (default is (0.25, 0.25, 0.25, 0.25)).
        
    itermax : int, optional
        Maximum number of iterations for the scoring algorithm (default is 1000).

    Returns
    -------
    idata_score : ndarray
        A 2D numpy array containing the interaction scores for each ligand-receptor pair.
        
    idata_direction : ndarray
        A 2D numpy array indicating the direction (e.g., ligand or receptor dominance) 
        of each interaction.
        
    comm_network : dict
        A dictionary representing the combined communication network, mapping 
        ligand-receptor pairs to their scores.
        
    S : ndarray
        A 2D numpy array of gene expression values for ligands.
        
    D : ndarray
        A 2D numpy array of gene expression values for receptors.

    Notes
    -----
    - The scoring mechanism relies on the spatial proximity of cells, calculating distances 
      between pairs of cells to inform the scoring.
    - Ligand-receptor interactions are evaluated based on COT and their co-expression in the spatial 
      context of the cells.
    - The output includes normalized scores for interactions and a directionality indicator.

    Examples
    --------
    >>> interaction_scores, interaction_directions, network, S, D = score_ot(adata, lr_df, interface_meta, interface_cell_pair)
    """
    data_genes = set(adata.var_names)
    ligs = list(set(lr_df.iloc[:,0]).intersection(data_genes))
    recs = list(set(lr_df.iloc[:,1]).intersection(data_genes))
    A = np.inf * np.ones([len(ligs), len(recs)], float)
    for i in range(len(lr_df)):
        tmp_lig = lr_df.iloc[i][0]
        tmp_rec = lr_df.iloc[i][1]
        if tmp_lig in ligs and tmp_rec in recs:
            A[ligs.index(tmp_lig), recs.index(tmp_rec)] = 1.0
    A = A
    S = adata[:,ligs].to_df().to_numpy()
    D = adata[:,recs].to_df().to_numpy()

    M = np.zeros((adata.shape[0], adata.shape[0]))
    rows, cols = zip(*interface_cell_pair.T)
    M[rows, cols] = [np.finfo(np.float32).eps+np.linalg.norm(adata.obsm['spatial'][p[0]] - adata.obsm['spatial'][p[1]]) for p in interface_cell_pair.T]
    M += M.T
    cutoff = float( np.max(np.max(M))* 1.1) * np.ones_like(A)
    M[M==0] = np.max(np.max(M)) * 10

    cot_eps = 0.11
    cot_rho = 1e1
    cot_weights = weight
    cot_nitermax = itermax

    comm_network = cot_combine_sparse(S, D, A, M, cutoff, \
                eps_p=cot_eps, eps_mu=cot_eps, eps_nu=cot_eps, rho=cot_rho, weights=cot_weights, nitermax=cot_nitermax)
    
    idata_score = pd.DataFrame(np.zeros((len(interface_cell_pair[0]), len(comm_network))), columns = lr_df.index)
    idata_direction = pd.DataFrame(np.ones((len(interface_cell_pair[0]), len(comm_network))), columns = lr_df.index) * -1
    for key, val in comm_network.items():
        l, r = ligs[key[0]], recs[key[1]]
        pair_name = f'{l}_{r}'
        vals = []
        dirs = []
        lsum = S[:, key[0]]
        rsum = D[:, key[1]]
        vals_coexp = []
        for pair in interface_cell_pair.T:
            coexp = np.sqrt(np.max([lsum[pair[0]]*rsum[pair[1]], lsum[pair[1]]*rsum[pair[0]]]))
            # coexp = np.sqrt(np.max([lsum[pair[0]]*rsum[pair[1]], lsum[pair[1]]*rsum[pair[0]]]))
            val_two = [val[pair[0], pair[1]], val[pair[1], pair[0]]]
            vals.append(np.max(val_two))
            vals_coexp.append(coexp)
            dirs.append(np.argmax(val_two))
        max_scale = np.max([np.max(vals), np.max(vals_coexp)])
        vals_coexp = (vals_coexp - np.min(vals_coexp)) / (np.max(vals_coexp) - np.min(vals_coexp))
        vals = (vals - np.min(vals)) / (np.max(vals) - np.min(vals))
        idata_score[pair_name] = np.maximum(vals_coexp, vals) * max_scale
        idata_direction[pair_name] = dirs
    return idata_score.to_numpy(), idata_direction.to_numpy(), comm_network, S, D


def score_ot_entropy(adata, lr_df, interface_meta, interface_cell_pair, weight=(0.25, 0.25, 0.25, 0.25), itermax = 1000):
    ot_score, direction, comm_network, S_smth, D_smth = score_ot(adata, lr_df, interface_meta, interface_cell_pair, weight, itermax)
    coexp_score, _ = score(adata, lr_df, interface_cell_pair, interface_meta)
    log_Aa = np.log1p((coexp_score + 0.00001)/(ot_score+0.00001))
    nonzero_mean = np.mean(ot_score, axis=0)
    alogAa = (ot_score + nonzero_mean)*log_Aa / 2
    return alogAa, direction, comm_network, S_smth, D_smth

def score_ot_weighted(adata, lr_df, interface_meta, interface_cell_pair, weight=(0.25, 0.25, 0.25, 0.25), itermax = 1000, alpha=0.01):
    ot_score, direction, comm_network, S_smth, D_smth = score_ot(adata, lr_df, interface_meta, interface_cell_pair, weight, itermax)
    coexp_score, _ = score(adata, lr_df, interface_cell_pair, interface_meta)
    return alpha*coexp_score + (1-alpha)*ot_score, direction, comm_network, S_smth, D_smth


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

def score_v1(adata, lr_df, pairs):
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
    
def find_interfaces(adata, cluster_key, lr_df, cutoff=None, is_sc=False):
    """
    Identify interfaces between cells based on interaction capacity and spatial distance.

    This function locates interfaces between cells by analyzing interaction capacity and 
    their spatial distances. It applies filtering based on a specified cutoff distance and 
    can optionally include self-interaction pairs for bulk ST data.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object containing spatial and expression data.

    cluster_key : str
        Key in `adata.obs` that specifies the clustering information for the cells.

    lr_df : DataFrame
        DataFrame containing ligand-receptor pairs to be analyzed.

    cutoff : float, optional
        Distance cutoff for filtering interfaces (default is None). If not specified, 
        the cutoff is determined based on the distribution of distances.

    is_sc : bool, optional
        If True, self-interaction pairs are not included (default is False).

    Returns
    -------
    pairs : ndarray
        Array of identified cell pairs representing interfaces.

    pairs_meta : DataFrame
        DataFrame containing metadata for the identified interfaces, including distances.

    Notes
    -----
    - The function calculates a median distance and uses it to filter out low-capacity cells 
      and their interfaces.
    - Gaussian Mixture Models (GMM) are used to estimate the ideal number of interfaces 
      based on the capacity of the cells.
    - If `is_sc` is False, self-interaction pairs are added to the results.

    Examples
    --------
    >>> pairs, pairs_meta = find_interfaces(adata, 'cluster_key', lr_df)
    """
    pairs = power_tri_init(adata, lr_df)
    pairs_meta = meta(adata, cluster_key, pairs)
    
    # filter by distance
    if not cutoff:
        median = np.median(pairs_meta['dist'])
        min_dist = np.min(pairs_meta['dist'])
        if median == min_dist:
            print('using 0.75')
            cutoff = np.quantile(pairs_meta['dist'], 0.75)
        else:
            print('using 0.99')
            cutoff = np.quantile(pairs_meta['dist'], 0.99)
    org_number = len(pairs_meta)
    pairs = pairs[:, pairs_meta['dist'] <= cutoff]
    pairs_meta = pairs_meta[pairs_meta['dist'] <= cutoff]
    print(f'Located {len(pairs_meta)} interfaces on {org_number} power cell boundaries with distance cutoff {cutoff}.')
    
    capacity = (adata.obs['capacity'] / max(adata.obs['capacity'])).to_numpy().reshape(-1,1)
    val, count = np.unique(pairs_meta[['A', 'B']].to_numpy().flatten(), return_counts=True)
    gmm = GaussianMixture(n_components=np.max(count), random_state=0).fit(capacity)
    gmm_labels = gmm.predict(X=capacity / max(capacity))
    gmm_labels = np.argsort(gmm.means_.flatten())[gmm_labels]
    adata.obs['n_interface'] = 0
    adata.obs.loc[val, 'n_interface'] = count
    adata.obs['ideal_n_interface'] = gmm_labels + 1
    median = np.median(pairs_meta['dist'])
    
    to_remove = []
    for to_prune_class in range(5):
        to_prune = adata.obs_names[np.argwhere(gmm_labels == to_prune_class)].flatten()
        for i in to_prune:
            interfaces = pairs_meta[(pairs_meta['A'] == i) | (pairs_meta['B'] == i)].sort_values(by='dist')
            interfaces = interfaces[interfaces['dist'] > median]
            to_remove += list(interfaces.index[1:])
    to_remove = np.unique(to_remove)
    print(f'Dropped {len(to_remove)} out of {len(pairs_meta)} interfaces for low capacity cells.')
    pairs = pairs[:, pairs_meta.index.isin(to_remove)==False]
    pairs_meta = pairs_meta.drop(to_remove)

    if not is_sc:
        # add i-i pairs
        self_pairs = [range(adata.shape[0]), range(adata.shape[0])]
        self_pairs_meta = meta(adata, cluster_key, self_pairs)
        pairs = np.concatenate((pairs, self_pairs), axis=1)
        pairs_meta = pd.concat([pairs_meta, self_pairs_meta], axis=0)
        
    return pairs, pairs_meta

def subset_adata(adata, lr_df, pathway_df, imputation, normalize_total):
    if imputation:
        print('Running imputation with MAGIC')
        impute_MAGIC(adata)
    
    genes = adata.var_names.tolist()
    # get lr genes
    lr_df = lr_df[lr_df['ligand'].isin(genes) & lr_df['receptor'].isin(genes)]
    lr_df.index = lr_df['ligand'] + "_" + lr_df['receptor']
    l = lr_df['ligand'].to_numpy().flatten()
    r = lr_df['receptor'].to_numpy().flatten()
    unique_lr = np.unique(np.concatenate((l, r)))
    # get pathway tf genes
    pathway_df = subset_pathway_df(pathway_df, lr_df, adata)
    unique_tf = np.unique(pathway_df[['src', 'dest']])
    # subset adata
    adata = adata[:, adata.var_names.isin(np.concatenate((unique_lr, unique_tf)))]
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.filter_cells(adata, min_genes=1)
    # normalize total will tenper the expression built by simulation
    if normalize_total:
        sc.pp.normalize_total(adata)
        print('Normalizing total counts per cell to the median of total counts for cells before normalization')
        # sc.pp.normalize_total(adata, target_sum=1e4)
    genes = adata.var_names.tolist()
    lr_df = lr_df[lr_df['ligand'].isin(genes) & lr_df['receptor'].isin(genes)]
    pathway_df = pathway_df[pathway_df['src'].isin(genes) & pathway_df['dest'].isin(genes)]
    return lr_df, pathway_df, adata


def find_pairs_v1(adata, coord_type='generic', n_neighs=6):
    from squidpy.gr import spatial_neighbors
    from scipy.sparse import triu
    if coord_type == 'grid':
        spatial_neighbors(adata, coord_type=coord_type, n_neighs=n_neighs)
    else:
        spatial_neighbors(adata, coord_type=coord_type, delaunay=True, n_neighs=n_neighs)
    return np.transpose(triu(adata.obsp['spatial_connectivities']).nonzero()).T

def power_tri_init(adata, lr_df):
    """
    Initialize power triangulation based on interaction capacity and spatial data.

    This function computes the power triangulation for a set of cells based on their 
    spatial coordinates and ligand-receptor (LR) capacities. It normalizes the capacities, 
    computes lifted weighted points, and determines the convex hull and Delaunay triangulation 
    to identify interactions.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object containing spatial coordinates and expression data.

    lr_df : DataFrame
        DataFrame containing ligand-receptor pairs to be analyzed.

    Returns
    -------
    pairs : ndarray
        Array of unique pairs of cell indices representing power triangulation edges.

    Notes
    -----
    - The function normalizes the ligand-receptor capacity per cell and computes a minimum 
      distance based on spatial coordinates.
    - The lifted weighted points are calculated from the spatial data and capacities.
    - The convex hull is computed to extract the Delaunay triangulation, which is then used 
      to generate unique cell pairs representing interfaces.

    Examples
    --------
    >>> pairs = power_tri_init(adata, lr_df)
    """
    # normalize lr capacity per cell
    unique_lr = np.unique(lr_df[['ligand', 'receptor']].to_numpy().flatten())
    adata_exp = adata.to_df()
    adata_exp = adata_exp / adata_exp.sum(axis=0)
    capacity = adata_exp[unique_lr].sum(axis=1).values
    min_dist = np.min(pdist(adata.obsm['spatial'][:, ::-1]))
    adata.obs['capacity'] = capacity
    capacity = (capacity - np.min(capacity)) / (np.max(capacity) - np.min(capacity)) + 0.01 
    capacity = capacity * min_dist
	# Compute the lifted weighted points
    s_norm = np.sum(adata.obsm['spatial'] ** 2, axis = 1) - capacity ** 2
    s_lifted = np.concatenate([adata.obsm['spatial'], s_norm[:,None]], axis = 1)
	# Compute the convex hull of the lifted weighted points
    hull = ConvexHull(s_lifted)
	# Extract the Delaunay triangulation from the lower hull
    tri_list = tuple([a, b, c] if is_ccw_triangle(adata.obsm['spatial'][a], adata.obsm['spatial'][b], adata.obsm['spatial'][c]) else [a, c, b]  for (a, b, c), eq in zip(hull.simplices, hull.equations) if eq[2] <= 0)
	# Compute the Voronoi points
    return np.unique(list(list(sorted(edge)) for tri in tri_list for edge in itertools.combinations(tri, 2)), axis=0).T

def meta(adata, cluster_key, pairs):
    # get label
    pairs_meta = pd.DataFrame()
    pairs_meta['A'] = adata.obs_names[pairs[0]]
    pairs_meta['B'] = adata.obs_names[pairs[1]]
    pairs_meta[['A_row', 'A_col']] = adata.obsm['spatial'][pairs[0]]
    pairs_meta[['B_row', 'B_col']] =  adata.obsm['spatial'][pairs[1]]

    if cluster_key != '' and cluster_key in adata.obs.keys(): 
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

def norm2(X):
	return np.sqrt(np.sum(X ** 2))

def normalized(X):
	return X / norm2(X)

def get_triangle_normal(A, B, C):
	return normalized(np.cross(A, B) + np.cross(B, C) + np.cross(C, A))


def get_power_circumcenter(A, B, C):
	N = get_triangle_normal(A, B, C)
	return (-.5 / N[2]) * N[:2]

def is_ccw_triangle(A, B, C):
	M = np.concatenate([np.stack([A, B, C]), np.ones((3, 1))], axis = 1)
	return np.linalg.det(M) > 0