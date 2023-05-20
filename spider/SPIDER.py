import pandas as pd


from . import svi
from . import preprocess
from . import clustering
from . import enrichment
from . import visualization
from . import util
from . import trajectory

class SPIDER():
    def __init__(self):
        self.svi = svi
        self.pp = preprocess
        self.cl = clustering
        self.er = enrichment
        self.vis = visualization
        self.util = util
        self.traj = trajectory
        pass

    def prep(self,
            adata_input, work_dir, 
            no_spatalk=False,
            cluster_key='type', 
            is_human=True, 
            n_neighs=6, 
            coord_type='grid', 
            imputation=True,
            overwrite=False,
    ):
        adata = adata_input.copy()
        del adata_input
        # detect cci

        # Step 1
        pairs = preprocess.find_pairs(adata, coord_type, n_neighs)
        pairs_meta = preprocess.meta(adata, cluster_key, pairs)
        # Step 2
        lr_raw = preprocess.subset_lr(adata, no_spatalk, work_dir, cluster_key, is_human, overwrite)
        lr_df, adata = preprocess.subset_adata(adata, lr_raw)
        score = preprocess.score(adata, lr_df, pairs, imputation)
        # Idata object construction
        idata = preprocess.idata_construct(score, pairs_meta, lr_df, lr_raw, adata)
        return idata

    def find_svi(self, idata, out_f, abstract=True, overwrite=False, n_neighbors=10, threshold=0.01, pattern_prune_threshold=0.0001):
        from os.path import exists
        from os import mkdir
        if not exists(out_f):
            print(f'Creating folder {out_f}')
            mkdir(out_f)
        if len(idata) < 200:
            print('number of interface is less than 200, skipping abstraction')
            abstract=False
        if abstract:
            som, idata, meta_idata = svi.abstract(idata, n_neighbors)
            svi.find_svi(meta_idata,out_f,overwrite, som=som) #generating results
            print('finished running all SVI tests')
            svi_df, svi_df_strict = svi.combine_SVI(meta_idata,threshold=threshold)
            if (overwrite) | (not exists(f'{out_f}pattern.csv')):
                svi.SVI_patterns(meta_idata, svi_df_strict, pattern_prune_threshold=pattern_prune_threshold)
                pd.DataFrame(meta_idata.obsm['pattern_score']).to_csv(f'{out_f}pattern.csv')
                meta_idata.var.to_csv(f'{out_f}membership.csv')
            else:
                meta_idata.obsm['pattern_score'] = pd.read_csv(f'{out_f}pattern.csv', index_col=0).to_numpy()
                meta_idata.var = pd.read_csv(f'{out_f}membership.csv', index_col=0)
            svi.meta_pattern_to_idata(idata, meta_idata)
            pd.DataFrame(meta_idata.obsm['pattern_score']).to_csv(f'{out_f}full_pattern.csv')
        else:
            svi.find_svi(idata, out_f,overwrite) #generating results
            svi_df, svi_df_strict = svi.combine_SVI(idata,threshold=threshold)
            if (overwrite) | (not exists(f'{out_f}pattern.csv')):
                svi.SVI_patterns(idata, svi_df_strict, pattern_prune_threshold=pattern_prune_threshold)
                pd.DataFrame(idata.obsm['pattern_score']).to_csv(f'{out_f}pattern.csv')
                idata.var.to_csv(f'{out_f}membership.csv')
            else:
                idata.obsm['pattern_score'] = pd.read_csv(f'{out_f}pattern.csv', index_col=0)
                idata.var = pd.read_csv(f'{out_f}membership.csv', index_col=0)   
        idata.var[[f'pattern_correlation_{x}' for x in range(idata.obsm['pattern_score'].shape[1])]] = 0
        corr_df=pd.concat([idata[:,idata.var['is_svi']==1].to_df(),pd.DataFrame(idata.obsm['pattern_score'],index=idata.obs_names)],axis=1).corr().loc[idata[:,idata.var['is_svi']==1].var_names, range(idata.obsm['pattern_score'].shape[1])]
        idata.var.loc[idata[:,idata.var['is_svi']==1].var_names, [f'pattern_correlation_{x}' for x in range(idata.obsm['pattern_score'].shape[1])]] = corr_df.to_numpy()
        return idata, None
        
    def cell_transform(self, idata, adata, label=None):
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
            print(f'Added key rank_interaction_score_groups, rank_interaction_pattern_groups in adata.uns')   
        adata.obsm['interaction_pattern'] = adata.obsm['interaction_pattern'].to_numpy()                                                   
        adata.obsm['interaction_score'] = adata.obsm['interaction_score'].to_numpy()                                                   
        return adata, adata_lri, adata_pattern
    




