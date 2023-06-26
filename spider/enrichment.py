import pandas as pd
import numpy as np

from .preprocess import load_lr_df

def load_pathway(is_human=True):
    from importlib import resources
    with resources.path("spider.lrdb", "pathway.tsv") as pw_fn:
        pw = pd.read_csv(pw_fn, sep='\t', index_col=0)
    species = 'Human' if is_human else 'Mouse'
    pw = pw[pw.species==species]
    return pw
    
def pathway_annotation(idata, is_human=True):
    pw = load_pathway(is_human)
    pw = pw.loc[pw.index.isin(idata.var_names)]
    df = pd.get_dummies(pw.pathway)
    df = df.groupby(df.index).sum()
    idata.varm['pathway'] = pd.DataFrame(0, columns=df.columns, index=idata.var_names)
    idata.varm['pathway'].loc[df.index] = df
    print(f'Added key pathway in idata.varm')

def pathway_annotation_list(LRI_list, is_human=True):
    pw = load_pathway(is_human)
    pw = pw.loc[pw.index.isin(LRI_list)]
    df = pd.get_dummies(pw.pathway)
    df = df.groupby(df.index).sum()
    return df

def pathway_prep(idata, is_human=True):
    lr_raw = load_lr_df(is_human=is_human)
    lr_raw.index = lr_raw.ligand + '_' + lr_raw.receptor
    pw = load_pathway(is_human)
    pw = pw.loc[np.intersect1d(lr_raw.index, pw.index)]
    custom = {}
    for ptw in pw.pathway.unique():
        custom[ptw] = pw[pw.pathway == ptw].index.tolist()
    background = np.intersect1d(idata.var_names, pw.index).tolist()
    return custom, background

def pathway_prep_custom_background(background, is_human=True):
    lr_raw = load_lr_df(is_human=is_human)
    lr_raw.index = lr_raw.ligand + '_' + lr_raw.receptor
    pw = load_pathway(is_human)
    pw.index = pw.src + '_' + pw.dest
    pw = pw.loc[np.intersect1d(lr_raw.index, pw.index)]
    custom = {}
    for ptw in pw.pathway.unique():
        custom[ptw] = pw[pw.pathway == ptw].index.tolist()
    background = np.intersect1d(background, pw.index).tolist()
    return custom, background

def pattern_enrichment_edge(idata, is_human=True, groupby='label', subset=[], order=[], custom=None, background=None):
    import gseapy
    if not (custom and background):
        custom, background = pathway_prep(idata, is_human=is_human)
    var = idata.var
    if len(subset) != 0:
        var = var[var[groupby].isin(subset)]
    arr = []
    for i in var.sort_values(groupby)[groupby].unique():
        lri_list = var[var[groupby] ==i].index.to_numpy().tolist()
        arr.append([str(i), lri_list])
    dfs = []
    for i in range(len(arr)):
        try:
            enr_res = gseapy.enrichr(gene_list=arr[i][1],
                            gene_sets=custom,
                            background=background)
                            # cutoff = cutoff)
            enr_res.res2d[groupby] = arr[i][0]
            dfs.append(enr_res.res2d)
        except Exception as e:
            print(e)
            continue
    merged_df = pd.concat(dfs)
    if len(order)!=0:
        from string import ascii_lowercase
        rename_dict = {}
        count = 0
        for x in order:
            rename_dict[str(x)] = f'({ascii_lowercase[count]}) {x}'
            count += 1
        merged_df[f'ordered_{groupby}'] = merged_df[groupby]
        merged_df[f'ordered_{groupby}'] = merged_df[groupby].map(rename_dict)
    else:
        merged_df[f'ordered_{groupby}'] = merged_df[groupby]
    return merged_df, arr

def pattern_enrichment_node(idata, is_human=True, groupby='label', subset=[], order=[], custom=None, background=None, custom_pathwaydb=[]):
    import gseapy
    var = idata.var
    if len(subset) != 0:
        var = var[var[groupby].isin(subset)]
    arr = []
    for i in var.sort_values(groupby)[groupby].unique():
        lri_list = var[var[groupby] ==i].index.to_numpy().tolist()
        gene_list = np.unique(np.concatenate([x.split('_') for x in lri_list])).tolist()
        arr.append([str(i), gene_list])
    if len(custom_pathwaydb) == 0:
        pathway_db = ['KEGG_2021_Human' if is_human else 'KEGG_2019_Mouse']
    else:
        pathway_db = custom_pathwaydb
    organism = 'Human' if is_human else 'Mouse'
    dfs = []
    for i in range(len(arr)):
        try:
            enr_res = gseapy.enrichr(gene_list=arr[i][1],
                            organism=organism,
                            gene_sets=pathway_db)
            enr_res.res2d[groupby] = arr[i][0]
            dfs.append(enr_res.res2d)
        except Exception as e:
            print(e)
            continue
    merged_df = pd.concat(dfs)
    if len(order)!=0:
        from string import ascii_lowercase
        rename_dict = {}
        count = 0
        for x in order:
            rename_dict[str(x)] = f'({ascii_lowercase[count]}) {x}'
            count += 1
        merged_df[f'ordered_{groupby}'] = merged_df[groupby]
        merged_df[f'ordered_{groupby}'] = merged_df[groupby].map(rename_dict)
    else:
        merged_df[f'ordered_{groupby}'] = merged_df[groupby]
    return merged_df, arr
    
    
def enrichment(custom, background, histology_results, groupby='pattern', cutoff=0.05, top=None, order=[], group=[]):
    import gseapy
    dfs=[]
    histology_results = histology_results[histology_results.g.isin(np.intersect1d(histology_results.g, background))]
    arr = []
    if len(group) == 0:
        for i in histology_results.sort_values(groupby)[groupby].unique():
            lri_list = histology_results[histology_results[groupby] ==i].sort_values('membership', ascending=False)['g'].to_numpy()
            if top:
                lri_list = lri_list[:top]
            lri_list = lri_list.tolist()
            arr.append([str(i), lri_list])
    else:
        count = 0
        for x in order:
            index_m  = np.where(group == x)[0].tolist()
            lri_lists = []
            for i in index_m:
                lri_lists.append(histology_results.query('@groupby == @i').sort_values('membership', ascending=False)['g'].tolist())
            lri_list = np.concatenate(lri_lists)
            if top:
                lri_list = lri_list[:top]
            lri_list = lri_list.tolist()
            arr.append([f'Merged groupby {count}: {index_m}', lri_list])
            count += 1

    for i in range(len(arr)):
        try:
            enr_res = gseapy.enrichr(gene_list=arr[i][1],
                            gene_sets=custom,
                            background=background,
                            cutoff = cutoff)
            enr_res.res2d['group'] = arr[i][0]
            dfs.append(enr_res.res2d)
        except:
            print(i)
            continue
    merged_df = pd.concat(dfs)
    if len(order)!=0 and len(group) == 0:
        from string import ascii_lowercase
        rename_dict = {}
        count = 0
        for x in order:
            rename_dict[str(x)] = f'({ascii_lowercase[count]}) {x}'
            count += 1
        merged_df['ordered_group'] = merged_df.group
        merged_df['ordered_group'] = merged_df.group.map(rename_dict)
    else:
        merged_df['ordered_group'] = merged_df.group
    return merged_df, arr

def enrichment_interacrtion_list(idata, lri_list, is_human=True):
    import gseapy
    custom, background=pathway_prep(idata, is_human=is_human)
    enr_res = gseapy.enrichr(gene_list=lri_list,
                            gene_sets=custom,
                            background=background)
    return enr_res.results

def enrichment_interacrtion_gene_list(idata, lri_list, is_human=True,custom_pathwaydb=[]):
    import gseapy
    gene_list = np.unique(np.concatenate([x.split('_') for x in lri_list])).tolist()
    if len(custom_pathwaydb) == 0:
        pathway_db = ['KEGG_2021_Human' if is_human else 'KEGG_2019_Mouse']
    else:
        pathway_db = custom_pathwaydb
    organism = 'Human' if is_human else 'Mouse'
    enr_res = gseapy.enrichr(gene_list=gene_list,
                    organism=organism,
                    gene_sets=pathway_db)
    return enr_res.results

def enrichment_interacrtion_gene_df(lri_df, groupby='label',is_human=True,custom_pathwaydb=[], order=[]):
    import gseapy
    arr = []
    for i in lri_df.sort_values(groupby)[groupby].unique():
        lri_list = lri_df[lri_df[groupby] ==i].index.to_numpy().tolist()
        gene_list = np.unique(np.concatenate([x.split('_') for x in lri_list])).tolist()
        arr.append([str(i), gene_list])
    if len(custom_pathwaydb) == 0:
        pathway_db = ['KEGG_2021_Human' if is_human else 'KEGG_2019_Mouse']
    else:
        pathway_db = custom_pathwaydb
    organism = 'Human' if is_human else 'Mouse'
    dfs = []
    for i in range(len(arr)):
        try:
            enr_res = gseapy.enrichr(gene_list=arr[i][1],
                            organism=organism,
                            gene_sets=pathway_db)
            enr_res.res2d[groupby] = arr[i][0]
            dfs.append(enr_res.res2d)
        except Exception as e:
            print(e)
            continue
    merged_df = pd.concat(dfs)
    if len(order)!=0:
        from string import ascii_lowercase
        rename_dict = {}
        count = 0
        for x in order:
            rename_dict[str(x)] = f'({ascii_lowercase[count]}) {x}'
            count += 1
        merged_df[f'ordered_{groupby}'] = merged_df[groupby]
        merged_df[f'ordered_{groupby}'] = merged_df[groupby].map(rename_dict)
    else:
        merged_df[f'ordered_{groupby}'] = merged_df[groupby]
    return merged_df, arr


def add_diff(idata, list1, list2, key_to_add):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    exp1=scaler.fit_transform(np.sqrt(idata[:, idata.var_names.isin(list1)].to_df()).sum(axis=1).to_numpy().reshape(-1, 1)).flatten()
    exp2=scaler.fit_transform(np.sqrt(idata[:, idata.var_names.isin(list2)].to_df()).sum(axis=1).to_numpy().reshape(-1, 1)).flatten()
    idata.obs[key_to_add]=exp1-exp2
    print(f'Added key {key_to_add} in idata.uns')