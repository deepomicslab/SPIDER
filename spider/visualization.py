import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
from statannotations.Annotator import Annotator
from .preprocess import load_lr_df
from .util import compare_var

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
            plt.title('Pattern {}'.format(i) )
            # plt.title('Pattern {} - {} LRIs'.format(i, np.sum(idata.var.label == i)) )
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
    # print(figsize)
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
        
def ct_pie(idata, svi, use_direction=True, drop_self=False):
    df = pd.concat([idata.to_df()[svi],  idata.obs], axis=1)
    key = 'label'
    if use_direction:
        df['direction'] = np.array(idata[:, idata.var_names==svi].layers['direction']).flatten()
        df['arrow'] = df.apply(lambda x: f'{x.A_label}->{x.B_label}' if x.direction==0 else f'{x.B_label}->{x.A_label}', axis=1)
        key = 'arrow'
    if drop_self:
        df = df[df.A_label != df.B_label]
    ax = df.groupby(key)[svi].mean().sort_values(ascending=False).plot.pie(colormap='tab20', ylabel='', title=svi, labeldistance=None)
    ax.legend(bbox_to_anchor=(1, 0.9), loc='upper left', frameon=False, fontsize=8)
    
def ct_pies(idata, svi, use_direction=True):
    plt.subplot(1, 4, 1)
    ct_pie(idata, svi, use_direction=True)
    plt.subplot(1, 4, 4)
    ct_pie(idata, svi, use_direction=True, drop_self=True)
    
def rotate_label(plot, element):
    # white_space = "                              "
    angles = plot.handles['text_1_source'].data['angle']
    characters = np.array(plot.handles['text_1_source'].data['text'])
    plot.handles['text_1_source'].data['text'] = np.array([x + ' ' * (len(x)*2 + 5) if x in characters[np.where((angles < -1.5707963267949) | (angles > 1.5707963267949))] else x for x in plot.handles['text_1_source'].data['text']])
    plot.handles['text_1_source'].data['text'] = np.array([' ' * (len(x)*2 + 5) + x if x in characters[np.where((angles > -1.5707963267949) | (angles < 1.5707963267949))] else x for x in plot.handles['text_1_source'].data['text']])
    angles[np.where((angles < -1.5707963267949) | (angles > 1.5707963267949))] += 3.1415926535898
    plot.handles['text_1_glyph'].text_align = "center"

def vis_ct_chord(idata, svi):
    import holoviews as hv
    from holoviews import opts, dim
    df = pd.concat([idata.to_df()[svi],  idata.obs], axis=1)
    df['direction'] = np.array(idata[:, idata.var_names==svi].layers['direction']).flatten()
    df['arrow'] = df.apply(lambda x: f'{x.A_label}->{x.B_label}' if x.direction==0 else f'{x.B_label}->{x.A_label}', axis=1)
    input_df = df.groupby('arrow')[svi].mean().reset_index(drop=False).query(f'{svi} > 0')
    input_df[['source', 'target']] = input_df['arrow'].str.split('->', expand=True).to_numpy() 
    node_df = pd.DataFrame(index=np.unique(np.concatenate((input_df['source'].to_numpy(), input_df['target'].to_numpy()))))
    node_df['group'] = 1
    links_df = input_df[['source', 'target', svi]]
    links_df.columns = ['source', 'target', 'value']

    hv.extension('bokeh')
    hv.output(size=200)

    nodes = hv.Dataset(node_df, 'index')

    chord = hv.Chord((links_df, nodes))
    chord.opts(
        opts.Chord(cmap='Set2', edge_cmap='Set2', edge_color=dim('source').str(), 
                labels='index', node_color=dim('index').str(),   hooks=[rotate_label]),
     )
    return chord

def compare_ot_coexp_v1(idata_ot, idata_coexp, title=''):
    var_ot = idata_ot.to_df().var(axis=0)
    mean_ot = idata_ot.to_df().mean(axis=0)
    sd_ot = idata_ot.to_df().std(axis=0)
    cv_ot = var_ot/mean_ot
    cv2_ot = var_ot/(mean_ot**2)

    var_raw = idata_coexp.to_df().var(axis=0)
    mean_raw = idata_coexp.to_df().mean(axis=0)
    sd_raw = idata_coexp.to_df().std(axis=0)
    cv_raw = var_raw/mean_raw
    cv2_raw = var_raw/(mean_raw**2)

    box_df_ot = pd.DataFrame({'var':var_ot, 'sd':sd_ot, 'cv':cv_ot, 'cv2':cv2_ot}, index = idata_ot.var_names).melt()
    box_df_ot['formulation'] = 'Optimal Transport'
    box_df_raw = pd.DataFrame({'var':var_raw, 'sd':sd_raw, 'cv':cv_raw, 'cv2':cv2_raw}, index = idata_coexp.var_names).melt()
    box_df_raw['formulation'] = 'Co-expression'

    box_df = pd.concat([box_df_ot, box_df_raw])
    box_df['value (log-scaled)'] = np.log(box_df['value'])
    box_df.columns = ['metric', 'value', 'formulation', 'value (log-scaled)']
    # box_df = box_df[box_df.metric.isin(['cv', 'cv2'])]
    # box_df = box_df[box_df.metric != 'sd' & box_df.metric != 'var']
    ax = sns.boxplot(data=box_df, x='metric', y='value (log-scaled)', hue='formulation',
                     hue_order=['Optimal Transport', 'Co-expression'], showfliers=False, palette=['#80b1d3', '#fb8072'])
    pairs = []
    # for i in ['cv', 'cv2']:
    for i in ['var', 'sd', 'cv', 'cv2']:
        pairs.append( ((i, 'Optimal Transport'), (i, 'Co-expression')))
    annot = Annotator(ax, pairs, data=box_df, x='metric',y='value (log-scaled)', 
                      hue='formulation', hue_order=['Optimal Transport', 'Co-expression'])
    # annot.configure(test='Mann-Whitney-gt',comparisons_correction="BH", correction_format="replace")
    annot.configure(test='Mann-Whitney-ls',comparisons_correction="BH", correction_format="replace")
    annot.apply_test()
    annot.annotate()
    if title != '':
        plt.title(title)
    plt.show()
    
def compare_ot_coexp(diff_local, title=''):
    ax = sns.boxplot(data=diff_local, x='n_neighbor', y='diff', hue='formulation',
                     hue_order=['Optimal Transport', 'Co-expression'], showfliers=False, palette=['#80b1d3', '#fb8072'])
    pairs = []
    # for i in ['cv', 'cv2']:
    for i in diff_local.n_neighbor.unique():
        pairs.append( ((i, 'Optimal Transport'), (i, 'Co-expression')))
    annot = Annotator(ax, pairs, data=diff_local, x='n_neighbor',y='diff', 
                      hue='formulation', hue_order=['Optimal Transport', 'Co-expression'])
    # annot.configure(test='Mann-Whitney-gt',comparisons_correction="BH", correction_format="replace")
    annot.configure(test='Mann-Whitney-ls',comparisons_correction="BH", correction_format="replace")
    annot.apply_test()
    annot.annotate()
    if title != '':
        plt.title(title)
    plt.show()
    
def compare_interface_capacity(idata, idata_old, title='', save=''):
    n_interface, mae_v1, mae_v2, mse_v1, mse_v2, r2_v1, r2_v2, y_true, y_pred_v1, y_pred_v2 = compare_var(idata, idata_old)
    fig, axes = plt.subplots(1,4,figsize=(15,4))

    maes = [mae_v1, mae_v2]   
    mses = [mse_v1, mse_v2]
    r2s = [r2_v1, r2_v2]

    # MAE plot
    axes[0].bar(['v1','v2'], maes, color=['#fb8072', '#80b1d3'])
    axes[0].set_xticks(['v1','v2'])
    axes[0].set_xticklabels(['Delauney','Capacity']) 
    axes[0].plot(['v1','v2'], maes, color='#fb8072')
    axes[0].set_title('MAE')

    # MSE plot
    axes[1].bar(['v1','v2'], mses, color=['#fb8072', '#80b1d3'])  
    axes[1].set_xticks(['v1','v2'])
    axes[1].set_xticklabels(['Delauney','Capacity'])
    axes[1].plot(['v1','v2'], mses, color='#fb8072')  
    axes[1].set_title('MSE')

    # R2 plot  
    axes[2].bar(['v1','v2'], r2s, color=['#fb8072', '#80b1d3'])
    axes[2].set_xticks(['v1','v2'])  
    axes[2].set_xticklabels(['Delauney','Capacity'])
    axes[2].plot(['v1','v2'], r2s, color='#fb8072')
    axes[2].set_title('R2 Score')
    
    axes[3].plot(y_true, y_true - y_pred_v1, 'o', color='#fb8072', label='Delauney')
    axes[3].plot(y_true+0.3, y_true - y_pred_v2, 'o', color='#80b1d3', label='Capacity')  
    axes[3].axhline(y=0, color='black')
    axes[3].set_title('Residuals')
    axes[3].legend()
    
    
    if title == '':
        fig.suptitle('Metric comparisons')
    else:
        fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    
    if save != '':
        n_interface.to_csv(save+'n_interface.csv')
    
    return n_interface

def plot_top_tf(idata, lri, show_tf=5, spot_size=1):
    tfs_corr = idata.uns['tf_corr'].loc[lri].sort_values(ascending=False)[:show_tf]
    tfs_corr = tfs_corr[tfs_corr!=0]
    if len(tfs_corr) == 0:
        print('no correlated TF')
    else:
        cols = [f'{x[0]}\ncorr={x[1]}' for x in tfs_corr.reset_index().to_numpy()]
        col_names = [lri+'-'+tf for tf in tfs_corr.index]
        idata.obs[cols] = idata.obsm['tf_score'][col_names]
        cols = [lri] + cols
        sc.pl.spatial(idata, color=cols, spot_size=spot_size, ncols=len(cols))
        
def pattern_chord(idata, pattern):
    import holoviews as hv
    from holoviews import opts, dim
    df = pd.concat([idata.to_df()[svi],  idata.obs], axis=1)
    df['direction'] = np.array(idata[:, idata.var_names==svi].layers['direction']).flatten()
    df['arrow'] = df.apply(lambda x: f'{x.A_label}->{x.B_label}' if x.direction==0 else f'{x.B_label}->{x.A_label}', axis=1)
    input_df = df.groupby('arrow')[svi].mean().reset_index(drop=False).query(f'{svi} > 0')
    input_df[['source', 'target']] = input_df['arrow'].str.split('->', expand=True).to_numpy() 
    node_df = pd.DataFrame(index=np.unique(np.concatenate((input_df['source'].to_numpy(), input_df['target'].to_numpy()))))
    node_df['group'] = 1
    links_df = input_df[['source', 'target', svi]]
    links_df.columns = ['source', 'target', 'value']

    hv.extension('bokeh')
    hv.output(size=200)

    nodes = hv.Dataset(node_df, 'index')
    def rotate_label(plot, element):
        angles = plot.handles['text_1_source'].data['angle']
        characters = np.array(plot.handles['text_1_source'].data['text'])
        plot.handles['text_1_source'].data['text'] = np.array([x + ' ' * int(len(x)) if x in characters[np.where((angles < -1.5707963267949) | (angles > 1.5707963267949))] else x for x in plot.handles['text_1_source'].data['text']])
        new_text = []
        for x in plot.handles['text_1_source'].data['text']:
            new = x
            if x in characters[np.where((angles > -1.5707963267949) & (angles < 1.5707963267949))]:
                new = ' ' * int(len(x)) + new
            if x in characters[np.where((angles > -1.5707963267949) & (angles < 0))]:
                new = '\n\n' + new
            else:
                new = new + '\n\n'
            new_text.append(new)
        plot.handles['text_1_source'].data['text'] = np.array(new_text)
        angles[np.where((angles > 1.5707963267949))] = 0 # left top
        angles[np.where((angles < -1.5707963267949))] = 0  # left bottom
        angles[np.where((angles < 1.5707963267949)  & (angles > 0) )] = 0 # right top
        angles[np.where((angles > -1.5707963267949) & (angles < 0))] = 0 # right bottom
        plot.handles['text_1_glyph'].text_align = "center"
    chord = hv.Chord((links_df, nodes))
    chord.opts(
        opts.Chord(cmap='Set2', edge_cmap='Set2', edge_color=dim('source').str(), 
                labels='index', node_color=dim('index').str(),   hooks=[rotate_label], label_text_font_size='17pt', title=f'{svi}'),
    )
    return chord