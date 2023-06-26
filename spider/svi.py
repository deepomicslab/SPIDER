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
    for i in np.array(['SOMDE', 'SpatialDE', 'SPARKX', 'nnSVG', 'scGCO', 'gearyC', 'moranI'])[np.isin(['SOMDE', 'SpatialDE', 'SPARKX', 'nnSVG', 'scGCO', 'gearyC', 'moranI'],list(meta_idata.uns.keys()))]:
        idata.uns[i] = meta_idata.uns[i]
    print(f'Added key pattern_score in idata.obsm')   

def find_svi(idata, out_f, overwrite, R_path, som=None):
    # Method: Identifying spatially variable LR interactions
    # Gaussian models
    svi_nnSVG(idata,out_f,R_path,overwrite)
    svi_SOMDE(idata,out_f,overwrite, som=som)
    svi_SpatialDE2_omnibus(idata,out_f,overwrite)
    # svi_SpatialDE2(idata,out_f,overwrite)
    # Non-parametric covariance test
    svi_SPARKX(idata,out_f,R_path,overwrite)
    # HMRF
    svi_scGCO(idata,out_f,overwrite)
    # baseline auto-correlation metrics
    svi_moran(idata,out_f,overwrite)
    svi_geary(idata,out_f,overwrite)

def svi_moran(idata, work_dir, overwrite=False):
    try:
        t0=time.time()
        n_perms=1000
        if (overwrite) | (not exists( f'{work_dir}moranI.csv')):
            sq.gr.spatial_neighbors(idata, key_added='spatial')
            sq.gr.spatial_autocorr(
                idata,
                mode="moran",
                n_perms=n_perms,
                n_jobs=10,
            )
            idata.uns['moranI_time'] = time.time()-t0
            idata.uns['moranI'].to_csv(f'{work_dir}moranI.csv')
        result = pd.read_csv(f'{work_dir}moranI.csv', index_col=0)
        idata.uns['moranI'] = result
        print(f'Added key moranI in idata.uns')
    except:
        pass

def svi_geary(idata, work_dir,overwrite=False):
    try:
        t0=time.time()
        n_perms=1000
        if (overwrite) | (not exists( f'{work_dir}gearyC.csv')):
            sq.gr.spatial_neighbors(idata, key_added='spatial')
            sq.gr.spatial_autocorr(
                idata,
                mode="geary",
                n_perms=n_perms,
                n_jobs=10,
            )
            idata.uns['gearyC_time'] = time.time()-t0
            idata.uns['gearyC'].to_csv(f'{work_dir}gearyC.csv')
        result = pd.read_csv(f'{work_dir}gearyC.csv', index_col=0)
        idata.uns['gearyC'] = result
        print(f'Added key gearyC in idata.uns')
    except:
        pass

def svi_nnSVG(idata, work_dir, R_path, overwrite=False):
    try:
        count_f = f'{work_dir}idata_count.csv'
        meta_f = f'{work_dir}idata_meta.csv'
        if (overwrite) | ((not exists(count_f)) & (not exists(meta_f))):
            idata.to_df().to_csv(count_f)
            idata.obs[['row', 'col']].to_csv(meta_f)
        if (overwrite) | (not exists( f'{work_dir}nnSVG.csv')):
            t0=time.time()
            with resources.path("spider.R_script", "run_nnSVG.R") as pw_fn:
                os.system(str(f'/bin/bash -c "{R_path} -f {pw_fn} {count_f} {meta_f} {work_dir}"'))
            idata.uns['nnSVG_time'] = time.time()-t0
        result = pd.read_csv(f'{work_dir}nnSVG.csv', index_col=0)
        idata.uns['nnSVG'] = result
        print(f'Added key nnSVG in idata.uns')
    except Exception as e:
        print(e)
    
def scGCO_sv(locs, data_norm, cellGraph, gmmDict, smooth_factor=10, unary_scale_factor=100, label_cost=10, algorithm='expansion'):
    from itertools import repeat
    from functools import reduce
    import operator
    import statsmodels.stats.multitest as multi
    import scGCO

    results = [scGCO.compute_single_fixed_sf(locs, data_norm, cellGraph, gmmDict, w=None, n=None, smooth_factor=10, unary_scale_factor=100, label_cost=10, algorithm='expansion')]
    
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
    #exp_fdr = multi.multipletests(np.array(exp_pvalues), method='fdr_bh')[1]    
    
    labels_array = np.array(pred_labels).reshape(len(genes), pred_labels[0].shape[0])
    data_array = np.array((genes, p_values, fdr,s_factors, nodes, model_labels), dtype=object).T
    t_array = np.hstack((data_array, labels_array))
    c_labels = ['p_value', 'fdr',  'smooth_factor', 'nodes','model_labels']
    for i in np.arange(labels_array.shape[1]) + 1:
        temp_label = 'label_cell_' + str(i)
        c_labels.append(temp_label)
    result_df = pd.DataFrame(t_array[:,1:], index=t_array[:,0], columns=c_labels)
    return result_df

def svi_scGCO(idata, work_dir, overwrite=False):
    try:
        import scGCO
        if (overwrite) | (not exists( f'{work_dir}scGCO.csv')):
            t0=time.time()
            data_norm = scGCO.normalize_count_cellranger(idata.to_df())
            exp= data_norm.iloc[:,0]
            pos_key=['row', 'col']
            locs = idata.obs[pos_key].to_numpy()
            cellGraph= scGCO.create_graph_with_weight(locs, exp)
            gmmDict= scGCO.gmm_model(data_norm)
            result_df= scGCO_sv(locs, data_norm, cellGraph,gmmDict)
            scGCO.write_result_to_csv(result_df, f'{work_dir}scGCO.csv')
            idata.uns['scGCO_time'] = time.time()-t0
        result = pd.read_csv(f'{work_dir}scGCO.csv')
        idata.uns['scGCO'] = result[result.columns[:6]]
        print(f'Added key scGCO in idata.uns')
    except:
        pass
    
def svi_SPARKX(idata, work_dir, R_path, overwrite=False):
    try:
        count_f = f'{work_dir}idata_count.csv'
        meta_f = f'{work_dir}idata_meta.csv'
        if (overwrite) | ((not exists(count_f)) & (not exists(meta_f))):
            idata.to_df().to_csv(count_f)
            idata.obs[['row', 'col']].to_csv(meta_f)
        if (overwrite) | (not exists( f'{work_dir}SPARKX.csv')):
            t0=time.time()
            with resources.path("spider.R_script", "run_SPARKX.R") as pw_fn:
                os.system(str(f'/bin/bash -c "{R_path} -f {pw_fn} {count_f} {meta_f} {work_dir}"'))
            idata.uns['SPARKX_time'] = time.time()-t0
        result = pd.read_csv(f'{work_dir}SPARKX.csv', index_col=0)
        idata.uns['SPARKX'] = result
        print(f'Added key SPARKX in idata.uns')
    except:
        pass
    
def svi_SpatialDE2(idata, work_dir, overwrite=False):
    try:
        if (overwrite) | (not exists(f'{work_dir}SpatialDE.csv')):
            from spider import SpatialDE2
            t0=time.time()
            np.random.seed(20230617)
            svg_full, individual = SpatialDE2.test(idata, omnibus=False)
            svg_full = pd.concat([svg_full.set_index('gene'), individual.loc[individual.groupby('gene').lengthscale.idxmin()].set_index('gene')], axis=1)
            svg_full.to_csv(f'{work_dir}SpatialDE.csv')
            individual.to_csv(f'{work_dir}SpatialDE_individual.csv')
            idata.uns['SpatialDE2_time'] = time.time()-t0
        result = pd.read_csv(f'{work_dir}SpatialDE.csv', index_col=0)
        idata.uns['SpatialDE'] = result
        print(f'Added key SpatialDE in idata.uns')
    except:
        pass 
    
def svi_SpatialDE2_omnibus(idata, work_dir, overwrite=False):
    try:
        if (overwrite) | (not exists(f'{work_dir}SpatialDE_omnibus.csv')):
            from spider import SpatialDE2
            t0=time.time()
            svg_full, _ = SpatialDE2.test(idata, omnibus=True)
            svg_full = svg_full.set_index('gene')
            svg_full.to_csv(f'{work_dir}SpatialDE_omnibus.csv')
            idata.uns['SpatialDE2_time'] = time.time()-t0
        result = pd.read_csv(f'{work_dir}SpatialDE_omnibus.csv', index_col=0)
        idata.uns['SpatialDE'] = result
        print(f'Added key SpatialDE in idata.uns')
    except Exception as e:
        print(e)
        pass 

def svi_SOMDE(idata, work_dir, overwrite=False, som=None):
    try:
        if (overwrite) | (not exists(f'{work_dir}SOMDE.csv')):
            t0=time.time()
            if som is None:
                from somde import SomNode
                df = idata.to_df().T
                corinfo = idata.obs
                corinfo["total_count"]=df.sum(0)
                X=corinfo[['row','col']].values.astype(np.float32)
                som = SomNode(X,10)
                ndf,ninfo = som.mtx(df)
            # nres = som.norm()
            from scipy import optimize
            from somde import regress_out
            phi_hat, _ = optimize.curve_fit(lambda mu, phi: mu + phi * mu ** 2, som.ndf.mean(1), som.ndf.var(1))
            dfm = np.log(som.ndf + 1. / (2 * np.abs(phi_hat[0])))
            som.nres = regress_out(som.ninfo, dfm, 'np.log(total_count)').T
            result, SVnum =som.run()
            result.to_csv(f'{work_dir}SOMDE.csv')
            idata.uns['SOMDE_time'] = time.time()-t0
        result = pd.read_csv(f'{work_dir}SOMDE.csv', index_col=0)
        idata.uns['SOMDE'] = result
        print(f'Added key SOMDE in idata.uns')
    except:
        pass
    
def combine_SVI(idata, threshold, svi_number=10):
    svi_df, svi_df_strict = combine_SVI_strict(idata,threshold=threshold)
    if len(svi_df_strict) <= svi_number:
        print(f'Detected SVI number is less than {svi_number}, falling back to relaxed filtering.')
        svi_df, svi_df_strict = combine_SVI_Fisher(idata,threshold=threshold)
    if len(svi_df_strict) <= svi_number:
        print(f'Detected SVI number is less than {svi_number}, falling back to use SOMDE result only.')
        svi_df, svi_df_strict  = combine_SVI_somde(idata,threshold=threshold)
    return svi_df, svi_df_strict

def combine_SVI_Fisher(idata, threshold=0.05):
    from scipy.stats import combine_pvalues
    methods = np.array(['SOMDE', 'SpatialDE', 'SPARKX', 'nnSVG', 'scGCO', 'gearyC', 'moranI'])[np.isin(['SOMDE', 'SpatialDE', 'SPARKX', 'nnSVG', 'scGCO', 'gearyC', 'moranI'],list(idata.uns.keys()))]
    df = []
    for i in methods:
        if i == 'SOMDE':
            df.append(idata.uns[i].set_index('g')[['qval']].rename(columns = {'qval': i}))
        elif i == 'SpatialDE':
            df.append(idata.uns[i][['padj']].rename(columns = {'padj': i}))
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
    idata.var['is_svi'] = 0
    idata.var.loc[df_sub.index, 'is_svi'] = 1
    return df, df_sub

def combine_SVI_strict(idata, threshold=0.01):
    methods = np.array(['SOMDE', 'SpatialDE', 'SPARKX', 'nnSVG', 'scGCO', 'gearyC', 'moranI'])[np.isin(['SOMDE', 'SpatialDE', 'SPARKX', 'nnSVG', 'scGCO', 'gearyC', 'moranI'],list(idata.uns.keys()))]
    print(f'Using the results from SVI identification methods: {methods}')
    df = []
    for i in methods:
        if i == 'SOMDE':
            df.append(idata.uns[i].set_index('g')[['qval']].rename(columns = {'qval': i}))
        elif i == 'SpatialDE':
            df.append(idata.uns[i][['padj']].rename(columns = {'padj': i}))
        elif i == 'SPARKX':
            df.append(idata.uns[i][['adjustedPval']].rename(columns = {'adjustedPval': i}))
        elif i == 'nnSVG':
            df.append(idata.uns[i][['padj']].rename(columns = {'padj': i}))
        elif i == 'scGCO':
            df.append(idata.uns[i].set_index(idata.uns['scGCO'].columns[0])[['fdr']].rename(columns = {'fdr': i}))
    df = pd.concat(df, axis=1).fillna(1)
    df_sub = df[(df<threshold).all(axis=1)]
    print(f'{len(df_sub)}/{len(df)} SVIs identified (threshold={threshold}).')
    idata.uns['svi'] = df
    idata.var['is_svi'] = 0
    df_sub = df_sub.loc[np.intersect1d(df_sub.index, idata.var_names)]
    idata.var.loc[df_sub.index, 'is_svi'] = 1
    return df, df_sub

def combine_SVI_somde(idata, threshold=0.01):
    print(f'Using the SOMDE results')
    df = idata.uns['SOMDE'].set_index('g')[['qval']].rename(columns = {'qval': 'SOMDE'}).fillna(1)
    df_sub = df[(df<threshold).all(axis=1)]
    print(f'{len(df_sub)}/{len(df)} SVIs identified (threshold={threshold}).')
    idata.uns['svi'] = df
    idata.var['is_svi'] = 0
    idata.var.loc[df_sub.index, 'is_svi'] = 1
    return df, df_sub
    
def eva_SVI(idata, svi_df_strict):
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
        ax =sns.boxplot(data=normalized_df,x='Metric',y='value', hue='Category', palette={'SVI':'#80b1d3','Excluded': '#fb8072'}, width=0.8, hue_order=['SVI', 'Excluded'])
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
        
def eva_pattern(idata):
    import seaborn as sns
    from statannotations.Annotator import Annotator
    
    dfs = []
    pairs = []
    for i in range(idata.obsm['pattern_score'].shape[1]):
        subdf = idata.var[[f'pattern_membership_{i}', f'pattern_correlation_{i}']]
        subdf.columns = ['membership', 'correlation']
        subdf['pattern'] = i
        dfs.append(subdf)
        pairs.append(((i, 'member'), (i, 'non-member')))
    maindf = pd.concat(dfs)
    maindf = maindf.sort_values('membership')
    maindf['membership'] = maindf['membership'].replace(0, 'non-member').replace(1, 'member')
    ax=sns.boxplot(data=maindf, x='pattern', y='correlation', hue='membership', hue_order=['member', 'non-member'],
                   palette={'member':'#80b1d3','non-member': '#fb8072'}, width=0.8)
    ax.legend(loc='upper center',ncol=2, bbox_to_anchor=(0.5, 1.1), frameon=False)
    annot = Annotator(ax, pairs, data=maindf, x='pattern',y='correlation', hue='membership')
    annot.configure(test='Mann-Whitney-ls',comparisons_correction="BH", correction_format="replace")
    annot.apply_test()
    annot.annotate()
    
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
    if (predefined_pattern_number == -1) & ((upper_patterns.patterns.shape[1] > 20) | (upper_patterns.patterns.shape[1] < 5)) :
        param_obj = SpatialDE2.SpatialPatternParameters(maxiter=iter, pattern_prune_threshold=pattern_prune_threshold, nclasses=20)
        upper_patterns, _ = SpatialDE2.spatial_patterns(idata, normalized=True, genes=allsignifgenes, rng=np.random.default_rng(seed=45), params=param_obj, copy=False)
    if (upper_patterns.patterns.shape[1] < 3) :
        param_obj = SpatialDE2.SpatialPatternParameters(maxiter=iter, pattern_prune_threshold=pattern_prune_threshold, nclasses=5)
        upper_patterns, _ = SpatialDE2.spatial_patterns(idata, normalized=True, genes=allsignifgenes, rng=np.random.default_rng(seed=45), params=param_obj, copy=False)
    print(f'eventually found {upper_patterns.patterns.shape[1]} patterns')
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
            histology_results['pattern'].to_csv('/home/lishiying/data6/01-interaction/results/DFPFC_paper/151673/patttern.csv')
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
    