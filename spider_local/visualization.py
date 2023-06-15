import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .preprocess import load_lr_df

def viz_interface_pattern(idata, pos_key=['row', 'col'], label='',  histology=None):
    plt.figure(figsize=(15, 15))
    if label != '':
        plt.subplot(6, 5, 1)
        plt.scatter(idata.uns['cell_meta'][pos_key[0]], idata.uns['cell_meta'][pos_key[1]], c=idata.uns['cell_meta'][label].astype("category").cat.codes, s=1)
        plt.axis('equal')
        base = 2
    else:
        base = 1
    for i in range(idata.uns['cell_pattern'].shape[1]):
        plt.subplot(6, 5, i + base)
        plt.scatter(idata.uns['cell_meta'][pos_key[0]], idata.uns['cell_meta'][pos_key[1]], c=idata.uns['cell_pattern'][i], s=1)
        plt.axis('equal')
        if histology:
            plt.title('Pattern {} - {} LRIs'.format(i, histology.query('pattern == @i').shape[0] ))
        plt.colorbar(ticks=[])
        
def pattern_LRI(idata, label='', pos_key=[], obsm_key='pattern_score', show_SVI=0,spot_size=1):
    plt.figure(figsize=(35, 50))
    if label != '':
        plt.subplot(20, 11, 1)
        ax=sns.scatterplot(data=idata.uns['cell_meta'], x=pos_key[0], y=pos_key[1], hue=label, s=20, palette ='tab20')
        # ax=sns.scatterplot(data=idata.obs, x='row', y='col', hue=label, s=10, palette ='tab20')
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1.05),ncol=int(len(idata.uns['cell_meta'][label].unique())/8+1), title=None, frameon=False)
        plt.axis('equal')
        plt.axis('off')
        base = 9
    else:
        base = 1
    if show_SVI==0:
        for i in range(idata.obsm[obsm_key].shape[1]):
            plt.subplot(20, 11, i + base)
            im=plt.scatter(idata.obs['row'], idata.obs['col'], c=idata.obsm[obsm_key][:,i], s=spot_size, cmap='plasma', edgecolors='none',linewidths=0)
            plt.axis('equal')
            plt.axis('off')
            plt.title('Pattern {} - {} LRIs'.format(i, np.sum(idata.var.label == i)) )
            plt.colorbar(im,fraction=0.046, pad=0.04)
    else:
        for i in range(idata.obsm[obsm_key].shape[1]):
            plt.subplot(20, 11, base)
            im=plt.scatter(idata.obs['row'], idata.obs['col'], c=idata.obsm[obsm_key][:,i], s=spot_size, cmap='plasma', edgecolors='none',linewidths=0)
            plt.axis('equal')
            plt.axis('off')
            plt.title('Pattern {} - {} LRIs'.format(i, np.sum(idata.var.label == i)))
            plt.colorbar(im,fraction=0.046, pad=0.04)
            base += 1
            svis = idata.var[idata.var.label == i].sort_values(f'pattern_correlation_{i}', ascending=False)
            svi_to_plot = svis.index.to_numpy()[:show_SVI]
            memberships = svis[f'pattern_correlation_{i}'].to_numpy()[:show_SVI]
            for j in range(show_SVI):
                if j < len(svi_to_plot):
                    plt.subplot(20,11, base)
                    im=plt.scatter(idata.obs['row'], idata.obs['col'], c=idata.to_df()[svi_to_plot[j]], s=spot_size, cmap='plasma', edgecolors='none',linewidths=0)
                    plt.axis('equal')
                    plt.axis('off')
                    plt.colorbar(im,fraction=0.046, pad=0.04)
                    plt.title(f'{svi_to_plot[j]}\n corr={"%.3f" % memberships[j]}')
                base += 1

def quiver_pattern(idata, label='', pos_key=[], obsm_key='pattern_score', traj_coords=[], show_SVI=0):
    plt.figure(figsize=(30, 50))
    if label != '':
        plt.subplot(20, 11, 1)
        ax=sns.scatterplot(data=idata.uns['cell_meta'], x=pos_key[0], y=pos_key[1], hue=label, s=10, palette ='tab20')
        # ax=sns.scatterplot(data=idata.obs, x='row', y='col', hue=label, s=10, palette ='tab20')
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1.05),ncol=int(len(idata.uns['cell_meta'][label].unique())/8+1), title=None, frameon=False)
        if len(traj_coords) != 0:
            for traj_cood in traj_coords:
                plt.quiver(traj_cood[0],traj_cood[2],traj_cood[1]-traj_cood[0],traj_cood[3]-traj_cood[2],color = 'red', angles='xy', scale_units='xy', scale = 1)
        plt.axis('equal')
        plt.axis('off')
        base = 9
    else:
        base = 1
    if show_SVI==0:
        for i in range(idata.obsm[obsm_key].shape[1]):
            plt.subplot(20, 11, i + base)
            im=plt.scatter(idata.obs['row'], idata.obs['col'], c=idata.obsm[obsm_key][:,i], s=1, cmap='plasma')
            if len(traj_coords) != 0:
                for traj_cood in traj_coords:
                    plt.quiver(traj_cood[0],traj_cood[2],traj_cood[1]-traj_cood[0],traj_cood[3]-traj_cood[2],color = 'red', angles='xy', scale_units='xy', scale = 1)
            plt.axis('equal')
            plt.axis('off')
            plt.title('Pattern {} - {} LRIs'.format(i, np.sum(idata.var.label == i)) )
            plt.colorbar(im,fraction=0.046, pad=0.04)
    else:
        for i in range(idata.obsm[obsm_key].shape[1]):
            plt.subplot(20, 11, base)
            im=plt.scatter(idata.obs['row'], idata.obs['col'], c=idata.obsm[obsm_key][:,i], s=1, cmap='plasma')
            if len(traj_coords) != 0:
                for traj_cood in traj_coords:
                    plt.quiver(traj_cood[0],traj_cood[2],traj_cood[1]-traj_cood[0],traj_cood[3]-traj_cood[2],color = 'red', angles='xy', scale_units='xy', scale = 1)
            plt.axis('equal')
            plt.axis('off')
            plt.title('Pattern {} - {} LRIs'.format(i, np.sum(idata.var.label == i)))
            plt.colorbar(im,fraction=0.046, pad=0.04)
            base += 1

            svis = idata.var[idata.var.label == i].sort_values(f'pattern_membership_{i}', ascending=False)
            svi_to_plot = svis.index.to_numpy()[:show_SVI]
            memberships = svis[f'pattern_membership_{i}'].to_numpy()[:show_SVI]
            for j in range(show_SVI):
                if j < len(svi_to_plot):
                    plt.subplot(20,11, base)
                    im=plt.scatter(idata.obs['row'], idata.obs['col'], c=idata.to_df()[svi_to_plot[j]], s=1, cmap='plasma')
                    if len(traj_coords) != 0:
                        for traj_cood in traj_coords:
                            plt.quiver(traj_cood[0],traj_cood[2],traj_cood[1]-traj_cood[0],traj_cood[3]-traj_cood[2],color = 'red', angles='xy', scale_units='xy', scale = 1)
                    plt.axis('equal')
                    plt.axis('off')
                    plt.colorbar(im,fraction=0.046, pad=0.04)
                    plt.title(f'{svi_to_plot[j]}')
                base += 1

def enrichment(df, x_key='ordered_group', figsize=None, size=None, save=None, cutoff=0.05, top_term=10):
    import gseapy
    df=df[df[x_key]!='-1']
    if not figsize:
        figsize = (3, int(top_term*len(df[x_key].unique())/10))
    if not size:
        size = int(top_term*len(df[x_key].unique())/20)
    ax = gseapy.dotplot(df,figsize=figsize, 
                    x=x_key, 
                    top_term=top_term,
                    column='Adjusted P-value',
                    cutoff=cutoff,
                    cmap = plt.cm.plasma, 
                    xticklabels_rot=30,
                    size=size, 
                    ofname=save,
                    show_ring=True)
    if not save:
        # ax.set_xlabel("")
        plt.show()

def svg_svi_relation(adata, idata, title, is_human=True, top=200):
    from matplotlib_venn import venn3,venn3_circles
    import gseapy
    
    organism = 'Human' if is_human else 'Mouse'
    pathway_db = 'KEGG_2021_Human' if is_human else 'KEGG_2019_Mouse'
    
    pathway_dfs=[]

    print(organism, pathway_db)

    genelist = adata.uns['moranI'][:top].index.to_list()
    enr_res_gene = gseapy.enrichr(gene_list=genelist,
                            cutoff=0.01,
                            organism=organism,
                            gene_sets=pathway_db)
    gene_pw_list = enr_res_gene.res2d[enr_res_gene.res2d['Adjusted P-value']<0.05].Term.to_numpy()
    enr_res_gene.res2d['group']='SVG'
    pathway_dfs.append(enr_res_gene.res2d[enr_res_gene.res2d['Adjusted P-value']<0.05])
    
    
    human_lr = load_lr_df(is_human)
    l = human_lr['ligand'].to_numpy().flatten()
    r = human_lr['receptor'].to_numpy().flatten()
    unique_lr_human = np.unique(np.concatenate((l, r)))
    genelist_lr = adata.uns['moranI'].loc[np.intersect1d(unique_lr_human,adata.uns['moranI'].index)].sort_values('I',ascending=False)[:top].index.to_list()
    print(len(genelist_lr))
    if len(genelist_lr)!=0:
        enr_res_gene = gseapy.enrichr(gene_list=genelist_lr,
                                cutoff=0.01,
                                organism=organism,
                                gene_sets=pathway_db)
        gene_lr_list = enr_res_gene.res2d[enr_res_gene.res2d['Adjusted P-value']<0.05].Term.to_numpy()
        enr_res_gene.res2d['group']='LR genes'
        pathway_dfs.append(enr_res_gene.res2d[enr_res_gene.res2d['Adjusted P-value']<0.05])
    else:
        gene_lr_list=[]

    interaction_list = idata.uns['moranI'][:top].index.to_list()
    gene_list_lri = np.unique(np.concatenate([x.split('_') for x in interaction_list])).tolist()
    enr_res = gseapy.enrichr(gene_list=gene_list_lri,
                                cutoff=0.01,
                                organism=organism,
                                gene_sets=pathway_db)
    lri_pw_list = enr_res.res2d[enr_res.res2d['Adjusted P-value']<0.05].Term.to_numpy()
    enr_res.res2d['group']='SVI genes'
    pathway_dfs.append(enr_res.res2d[enr_res.res2d['Adjusted P-value']<0.05])

    fig = plt.figure(figsize=(12, 5))
    fig.suptitle(title, fontsize=15)
    plt.subplot(1, 3, 1)
    moran_df = adata.uns['moranI'][:top]
    moran_df['label'] = 'SVG'
    moran_df_lr = adata.uns['moranI'].loc[genelist_lr]
    moran_df_lr['label'] = 'LR genes'
    moran_df_svi = adata.uns['moranI'].loc[np.intersect1d(gene_list_lri,adata.uns['moranI'].index)]
    moran_df_svi['label'] = 'SVI genes'
    df = pd.concat([moran_df, moran_df_lr,moran_df_svi])
    print(moran_df.I.mean(),moran_df_lr.I.mean(), moran_df_svi.I.mean())
    sns.boxplot(data=df, x="label", y="I", order=["SVI genes", "LR genes",  "SVG",], palette={"SVI genes":"#098154","LR genes":  "#069af3",  "SVG": "#c72e29"})
    plt.title('Moran I')
    
    plt.subplot(1, 3, 2)
    g=venn3(subsets = [set(gene_list_lri),set(genelist_lr),set(genelist)], 
        set_labels = ('SVI genes', "LR genes",'SVG'), 
        set_colors=("#098154","#069af3","#c72e29"),
        alpha=0.6,
        normalize_to=1.0,
       )
    plt.title('Gene Overlap')
    plt.subplot(1, 3, 3)
    g=venn3(subsets = [set(lri_pw_list),set(gene_lr_list),set(gene_pw_list)], 
        set_labels = ('SVI genes', "LR genes",'SVG'), 
        set_colors=("#098154","#069af3","#c72e29"),
        alpha=0.6,
        normalize_to=1.0,
       )
    plt.title('Enriched Pathway Overlap')
    
    merged_df = pd.concat(pathway_dfs)

    return merged_df,lri_pw_list,gene_lr_list,gene_pw_list

def paga_compare(idata, save=None):
    import scanpy as sc
    sc.pl.paga_compare(idata, legend_fontsize=5, frameon=False, size=20, legend_fontoutline=2, show=False)
    if save:
        plt.savefig(save, bbox_inches="tight",dpi=300)
        
    
def plot_projection(idata, save=None):
    import scvelo as scv
    scv.pl.velocity_embedding_stream(
        idata, color="label", vkey="T_fwd_spatial", basis="spatial", legend_loc="right",
        recompute=False,V=idata.obsm['T_fwd_spatial'],smooth=1.5,
        save=save
    )

def traj_proj(idata, label, save=''):
    import scvelo as scv
    if save == '':
        scv.pl.velocity_embedding_stream(
            idata, color=label, vkey="T_fwd_spatial", basis="spatial", legend_loc="right",
            recompute=False,V=idata.obsm['T_fwd_spatial'],smooth=1.5
        )
    else:
        scv.pl.velocity_embedding_stream(
            idata, color=label, vkey="T_fwd_spatial", basis="spatial", legend_loc="right",
            recompute=False,V=idata.obsm['T_fwd_spatial'],smooth=1.5,save=save,dpi=300
        )