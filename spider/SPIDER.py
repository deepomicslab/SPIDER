import pandas as pd
import numpy as np
from . import svi
from . import preprocess
from . import enrichment
from . import visualization
from . import util
from . import ot

class SPIDER():
    """
    The `SPIDER` class encapsulates methods for processing spatial transcriptomics data.

    The class provides functionality for constructing an interface data object from spatial transcriptomics (ST) data.
    It supports various preprocessing steps, such as filtering, normalization, and imputation, to prepare the data for
    downstream analysis of cell-cell interactions.
    """
    
    def __init__(self):
        self.svi = svi
        self.pp = preprocess
        self.er = enrichment
        self.vis = visualization
        self.util = util
        self.ot = ot
        pass

    def prep(self,
            adata_input, 
            cluster_key='type', 
            is_human=True, 
            cutoff=None,
            imputation=False,
            itermax = 1000,
            lr_raw = None,
            pathway_raw = None,
            is_sc=False,
            normalize_total = False,
        ):
        """
        Prepares the data for interface analysis by processing input AnnData.

        This method subsets the input AnnData object to include only ligand-receptor 
        genes and transcription factor genes, constructs interfaces from the data, 
        and computes interface profiles.

        Parameters
        ----------
        adata_input : AnnData
            Input AnnData object containing spatial transcriptomics data.
        cluster_key : str, optional
            Key in `adata` that denotes the cluster or cell type for analysis (default is 'type').
        is_human : bool, optional
            Indicates whether the input data is from human samples (default is True).
        cutoff : float, optional
            Threshold for filtering long-distance interactions (default is None, meaning no filtering).
        imputation : bool, optional
            If True, performs imputation on the input data (default is False).
        itermax : int, optional
            Maximum number of iterations for optimization algorithms (default is 1000).
        lr_raw : DataFrame, optional
            Predefined ligand-receptor pairs. If None, defaults to loading based on `is_human`.
        pathway_raw : DataFrame, optional
            Predefined pathways. If None, defaults to loading based on `is_human`.
        is_sc : bool, optional
            Indicates whether the input data is single-cell data (default is False).
        normalize_total : bool, optional
            If True, normalizes total counts in the `adata` object (default is False).

        Returns
        -------
        idata : AnnData
            An AnnData object constructed from the processed interface profiles and metadata.

        Notes
        -----
        - The method first creates a copy of the input AnnData object and releases the original.
        - It checks if ligand-receptor pairs and pathway data are provided; if not, it loads them based on the `is_human` flag.
        - Interfaces are constructed from the data, followed by scoring based on the provided parameters.
        - The resulting `idata` object contains computed scores and metadata for further analysis.

        Examples
        --------
        >>> idata = self.prep(adata_input, cluster_key='cell_type', is_human=True)
        """
        adata = adata_input.copy()
        del adata_input
        # Prep: find lr pairs and subset adata to have only lr genes and tf genes
        if lr_raw is None:
            lr_raw = preprocess.subset_lr(is_human)
        if pathway_raw is None:
            pathway_raw = preprocess.load_pathway_df(is_human)
        lr_df, pathway_df, adata = preprocess.subset_adata(adata, lr_raw, pathway_raw, imputation, normalize_total)
        # Step: construct interface
        interface_cell_pair, interface_meta = preprocess.find_interfaces(adata, cluster_key, lr_df, cutoff=cutoff, is_sc=is_sc)
        # Step: compute interface profile
        score, direction, _, _, _ = preprocess.score_ot(adata, lr_df, interface_meta, interface_cell_pair, itermax=itermax)
        # idata object construction
        idata = preprocess.idata_construct(score, direction, interface_meta, lr_df, lr_raw, pathway_df, adata, drop=False, normalize_total=normalize_total)
        return idata

    def find_svi(self, idata, out_f, R_path, abstract=True, overwrite=False, n_neighbors=5, alpha=0.3, threshold=0.01, pattern_prune_threshold=1e-20, predefined_pattern_number=-1, svi_number=10, n_jobs=10):
        """
        Find SVI and SVI patterns from interface idata.

        This method processes the provided interface data to compute SVI and related patterns. 
        It can optionally perform abstraction based on the number of interfaces present.

        Parameters
        ----------
        idata : AnnData
            Input AnnData object containing interface data.
        out_f : str
            Output directory path where results will be saved.
        R_path : str
            Path to the R script or binary for external processing.
        abstract : bool, optional
            If True, performs abstraction on the data if the number of interfaces is sufficient (default is True).
        overwrite : bool, optional
            If True, overwrites existing output files (default is False).
        n_neighbors : int, optional
            Number of neighbors to use for the abstraction process (default is 5).
        alpha : float, optional
            Alpha parameter for the abstraction algorithm (default is 0.3).
        threshold : float, optional
            Threshold for filtering SVI candidates (default is 0.01).
        pattern_prune_threshold : float, optional
            Threshold for pruning patterns during SVI pattern generation (default is 1e-20).
        predefined_pattern_number : int, optional
            If set to a positive integer, limits the number of predefined patterns (default is -1, which means no limit).
        svi_number : int, optional
            The number of SVI patterns to return (default is 10).
        n_jobs : int, optional
            Number of parallel jobs to run during processing (default is 10).

        Returns
        -------
        idata : AnnData
            The updated AnnData object containing results of SVI and SVI patterns.
        meta_idata : AnnData or None
            The meta AnnData object containing abstraction results, or None if abstraction was not performed.

        Notes
        -----
        - The method checks if the output directory exists and creates it if it doesn't.
        - If the number of interfaces in `idata` is less than 1000, abstraction is skipped.
        - Various intermediate results are saved as CSV files in the specified output directory.
        - The method utilizes external R processing for some computations, requiring R to be installed and accessible. Otherwise, only the three python methods are used.

        Examples
        --------
        >>> result_idata, result_meta_idata = op.find_svi(idata, 'output_directory', '/path/to/R', abstract=True)
        """
        from os.path import exists
        from os import mkdir
        
        if not exists(out_f):
            print(f'Creating folder {out_f}')
            mkdir(out_f)
        if len(idata) < 1000:
            print('number of interface is less than 2000, skipping abstraction')
            abstract=False
        if abstract:
            som, idata, meta_idata = svi.abstract(idata, n_neighbors, alpha)
            svi.find_svi(meta_idata, out_f, overwrite, R_path, som=som, n_jobs=n_jobs) #generating results
            print('finished running all SVI tests')
            if 'tf_support_count' in idata.var.columns:
                meta_idata.var['tf_support_count'] = idata.var['tf_support_count']
            svi_df, svi_df_strict = svi.combine_SVI(meta_idata, threshold=threshold, svi_number=svi_number)
            if (overwrite) | (not exists(f'{out_f}pattern.csv')):
                svi.SVI_patterns(meta_idata, svi_df_strict, pattern_prune_threshold=pattern_prune_threshold, predefined_pattern_number=predefined_pattern_number)
                pd.DataFrame(meta_idata.obsm['pattern_score']).to_csv(f'{out_f}pattern.csv')
                meta_idata.var.to_csv(f'{out_f}membership.csv')
            else:
                meta_idata.obsm['pattern_score'] = pd.read_csv(f'{out_f}pattern.csv', index_col=0).to_numpy()
                meta_idata.var = pd.read_csv(f'{out_f}membership.csv', index_col=0)
            svi.meta_pattern_to_idata(idata, meta_idata)
            pd.DataFrame(meta_idata.obsm['pattern_score']).to_csv(f'{out_f}full_pattern.csv')
        else:
            svi.find_svi(idata, out_f, overwrite, R_path, n_jobs=n_jobs) #generating results
            svi_df, svi_df_strict = svi.combine_SVI(idata, threshold=threshold, svi_number=svi_number)
            if (overwrite) | (not exists(f'{out_f}pattern.csv')):
                svi.SVI_patterns(idata, svi_df_strict, pattern_prune_threshold=pattern_prune_threshold, predefined_pattern_number=predefined_pattern_number)
                pd.DataFrame(idata.obsm['pattern_score']).to_csv(f'{out_f}pattern.csv')
                idata.var.to_csv(f'{out_f}membership.csv')
            else:
                idata.obsm['pattern_score'] = pd.read_csv(f'{out_f}pattern.csv', index_col=0).to_numpy()
                idata.var = pd.read_csv(f'{out_f}membership.csv', index_col=0)   
            meta_idata = None
        idata.var[[f'pattern_correlation_{x}' for x in range(idata.obsm['pattern_score'].shape[1])]] = 0
        corr_df=pd.concat([idata[:,idata.var['is_svi']==1].to_df(),pd.DataFrame(idata.obsm['pattern_score'],index=idata.obs_names)],axis=1).corr().loc[idata[:,idata.var['is_svi']==1].var_names, range(idata.obsm['pattern_score'].shape[1])]
        idata.var.loc[idata[:,idata.var['is_svi']==1].var_names, [f'pattern_correlation_{x}' for x in range(idata.obsm['pattern_score'].shape[1])]] = corr_df.to_numpy()
        if 'tf_corr' in idata.uns.keys():
            idata.uns['tf_corr'].replace(np.nan, 0, inplace=True)
        
        # svi.tf_pattern_to_idata(idata, idata_tf)
        
        return idata, meta_idata
    
    def find_svi_without_pattern(self, idata, out_f, R_path, abstract=True, overwrite=False, n_neighbors=5, alpha=0.3, threshold=0.01, pattern_prune_threshold=1e-6, predefined_pattern_number=-1, svi_number=10, n_jobs=10):
        from os.path import exists
        from os import mkdir

        if 'tf_support_count' not in idata.var.columns:
            print('not using tf info')
            idata.var['tf_support_count'] = 1
            idata.uns['tf_corr'] = pd.DataFrame()

        if not exists(out_f):
            print(f'Creating folder {out_f}')
            mkdir(out_f)
        if len(idata) < 1000:
            print('number of interface is less than 1000, skipping abstraction')
            abstract=False
        if abstract:
            som, idata, meta_idata = svi.abstract(idata, n_neighbors, alpha)
            svi.find_svi(meta_idata, out_f, overwrite, R_path, som=som, n_jobs=n_jobs, skip_metric=True) #generating results
            if 'tf_support_count' in idata.var.columns:
                meta_idata.var['tf_support_count'] = idata.var['tf_support_count']
            svi_df, svi_df_strict = svi.combine_SVI(meta_idata, threshold=threshold, svi_number=svi_number)
        else:
            svi.find_svi(idata, out_f, overwrite, R_path, n_jobs=n_jobs, skip_metric=True) #generating results
            svi_df, svi_df_strict = svi.combine_SVI(idata, threshold=threshold, svi_number=svi_number)
            meta_idata = None
        idata.uns['tf_corr'].replace(np.nan, 0, inplace=True)
        return idata, meta_idata
        
    def cell_transform(self, idata, adata, label=None):
        """
        Transform cell data by integrating interaction scores and patterns.

        This method updates the provided AnnData object (`adata`) with interaction patterns 
        and scores obtained from the input data (`idata`). It can also perform gene ranking 
        based on specified labels if provided.

        Parameters
        ----------
        idata : AnnData
            Input AnnData object containing interaction scores and patterns.

        adata : AnnData
            Input AnnData object to be updated with interaction data.

        label : str, optional
            The key for the label in `idata.uns['cell_meta']` used for grouping cells 
            and performing rank gene analysis (default is None).

        Returns
        -------
        adata : AnnData
            Updated AnnData object with added interaction patterns and scores.

        adata_lri : AnnData
            AnnData object containing interaction scores filtered by the specified label.

        adata_pattern : AnnData
            AnnData object containing interaction patterns filtered by the specified label.

        Notes
        -----
        - The function checks if the specified label exists in the `cell_meta` metadata. If it does, it computes ranking for genes based on interaction scores and patterns.
        - Small clusters (with only one cell) are excluded from the analysis to ensure meaningful group comparisons.
        - The spatial coordinates are preserved in the `obsm` attribute for both `adata_lri` and `adata_pattern`.

        Examples
        --------
        >>> adata, adata_lri, adata_pattern = cell_transform(idata, adata, label='cell_type')
        """
        from scanpy.tools import rank_genes_groups
        import anndata
        adata = adata[adata.obs_names.isin(idata.uns['cell_meta'].index)]
        util.scored_spot_interface(idata)
        util.interaction_spot_interface(idata)
        adata.obsm['interaction_pattern'] = idata.uns['cell_pattern'].loc[adata.obs_names]
        adata.obsm['interaction_score'] = idata.uns['cell_score'].loc[adata.obs_names]
        print(f'Added key interaction_pattern, interaction_score in adata.obsm')
        
        if label is not None:
            adata_lri = anndata.AnnData(idata.uns['cell_score'])
            idata.uns['cell_meta'][label] = idata.uns['cell_meta'][label].astype(str).astype('category')
            small_clust = idata.uns['cell_meta'][label].value_counts()[idata.uns['cell_meta'][label].value_counts()==1].index.to_numpy()
            adata_lri.obs = idata.uns['cell_meta']
            adata_lri = adata_lri[~adata_lri.obs[label].isin(small_clust),:]
            rank_genes_groups(adata_lri, groupby=label)
            adata.uns['rank_interaction_score_groups'] = adata_lri.uns['rank_genes_groups']
            adata_pattern = anndata.AnnData(idata.uns['cell_pattern'])
            adata_pattern.obs = idata.uns['cell_meta']
            adata_pattern = adata_pattern[~adata_pattern.obs[label].isin(small_clust),:]
            rank_genes_groups(adata_pattern, groupby=label)
            adata.uns['rank_interaction_pattern_groups'] = adata_pattern.uns['rank_genes_groups']   
                  
            adata_lri.obsm['spatial'] = pd.DataFrame(adata.obsm['spatial'], index=adata.obs_names).loc[adata_lri.obs_names].to_numpy()                                       
            adata_pattern.obsm['spatial'] = pd.DataFrame(adata.obsm['spatial'], index=adata.obs_names).loc[adata_pattern.obs_names].to_numpy()                                       
                                                          
            print(f'Added key rank_interaction_score_groups, rank_interaction_pattern_groups in adata.uns')   
        adata.obsm['interaction_pattern'] = adata.obsm['interaction_pattern'].to_numpy()                                                   
        adata.obsm['interaction_score'] = adata.obsm['interaction_score'].to_numpy()                                         
        return adata, adata_lri, adata_pattern
    
    # DEVELOPTAL FUNCTION

    def prep_exp(self,
            adata_input, 
            cluster_key='type', 
            is_human=True, 
            cutoff=None,
            imputation=False,
            lr_raw = None,
            is_sc=True,
            normalize_total = False,
    ):
        adata = adata_input.copy()
        del adata_input
        # Prep: find lr pairs and subset adata to have only lr genes
        lr_raw = preprocess.subset_lr(is_human)
        lr_df, adata = preprocess.subset_adata(adata, lr_raw, imputation)
        # Step: construct interface
        interface_cell_pair, interface_meta = preprocess.find_interfaces(adata, cluster_key, lr_df, cutoff=cutoff, is_sc=is_sc)
        # Step: compute interface profile
        score, direction = preprocess.score(adata, lr_df, interface_cell_pair, interface_meta)
        # idata object construction
        idata = preprocess.idata_construct(score, direction, interface_meta, lr_df, lr_raw, adata, drop=False, normalize_total=normalize_total)
        return idata
    

    def prep_compare(self,
            adata_input, 
            cluster_key='type', 
            is_human=True, 
            cutoff=None,
            imputation=False,
            itermax = 1000,
            lr_raw = None,
            is_sc=True,
            normalize_total = False,
    ):
        adata = adata_input.copy()
        del adata_input
        # Prep: find lr pairs and subset adata to have only lr genes
        if lr_raw is None:
            lr_raw = preprocess.subset_lr(is_human)
        lr_df, adata = preprocess.subset_adata(adata, lr_raw, imputation, normalize_total)
        # Step: construct interface
        interface_cell_pair, interface_meta = preprocess.find_interfaces(adata, cluster_key, lr_df, cutoff=cutoff, is_sc=is_sc)
        # Step: compute interface profile
        # score, direction = preprocess.score(adata, lr_df, interface_cell_pair, interface_meta)
        score, direction, _, _, _ = preprocess.score_ot(adata, lr_df, interface_meta, interface_cell_pair, itermax=itermax)
        # score, direction, _, _, _ = preprocess.score_ot_entropy(adata, lr_df, interface_meta, interface_cell_pair, itermax=itermax)
        # score, direction, _, _, _ = preprocess.score_ot_weighted(adata, lr_df, interface_meta, interface_cell_pair, itermax=itermax, alpha=alpha)
        # idata object construction
        idata = preprocess.idata_construct(score, direction, interface_meta, lr_df, lr_raw, adata, drop=False, normalize_total=normalize_total)

        score_old, direction_old = preprocess.score(adata, lr_df, interface_cell_pair, interface_meta)
        idata_old = preprocess.idata_construct(score_old, direction_old, interface_meta, lr_df, lr_raw, adata)
        return idata, idata_old


