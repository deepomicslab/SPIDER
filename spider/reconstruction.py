from scipy.stats.stats import pearsonr
import numpy as np
import pandas as pd

def score_s(xs, xs_p):
    co_exp = np.multiply(xs, xs_p)
    return np.maximum(co_exp[:int(len(xs)/2)], co_exp[int(len(xs_p)/2):])

def rev(idata, st):
    lr_df = pd.DataFrame([x.split('_') for x in idata.var_names], columns=['ligand', 'receptor'])
    l = lr_df['ligand'].to_numpy().flatten()
    r = lr_df['receptor'].to_numpy().flatten()
    sub_exp = st[np.concatenate((l, r))]
    rev_gene = np.concatenate((r, l))
    return rev_gene, sub_exp
    

def reconstruction(sc_label, st, st_meta_sc, sc_lib, idata, p, seed):
    idx_obj = {}
    picked_cells_arr = np.empty((0,8))
    for ct in sc_label.celltype.unique():
        idx_obj[ct] = sc_label[sc_label.celltype == ct].index.to_numpy().astype(str)
    interface_df = idata.to_df()
    rev_gene, sub_exp = rev(idata, st)
    for s in st.index:
        picked_cells = []
        max_corr = 0
        max_corrs = []
        st_meta_sc_sub = st_meta_sc.loc[s].reset_index().to_numpy()
        exp = st.loc[s]
        for iter in range(1000):
            np.random.seed(seed*2000+iter)
            idx_arr = [np.random.choice(idx_obj[ct], 1)[0] for ct in st_meta_sc_sub[:, 2]]
            spot_merged_exp = sc_lib.loc[idx_arr].sum()
            corr = pearsonr(spot_merged_exp, exp)[0]
            
            spot_merged_exp_sub = spot_merged_exp[rev_gene]
            ss = s.replace('_', '-')
            interface_sub = interface_df.loc[idata.obs.index[(idata.obs['A'] == ss) | (idata.obs['B'] == ss)]]
            corrs = []
            if len(interface_sub) != 0:
                for c in interface_sub.index:
                    neis = c.split('_')
                    nei = neis[0] if neis[1]==s else neis[1]
                    nei = nei.replace('-', '_')
                    exp_nei = sub_exp.loc[nei]
                    spot_merged_interface = score_s(spot_merged_exp_sub.to_numpy(), exp_nei.to_numpy())
                    corr_if = pearsonr(spot_merged_interface, interface_sub.loc[c])[0]
                    corrs.append(corr_if)  
                mean_corr_interface =  np.mean(corrs)
                mean_corr = mean_corr_interface*p+corr*(1-p)
            else:
                mean_corr_interface = np.nan
                mean_corr = corr
                

            if mean_corr > max_corr:
                max_corr = mean_corr
                max_corrs = [corr, mean_corr_interface]
                picked_cells = idx_arr
        st_meta_sc_sub = np.hstack([st_meta_sc_sub, np.array(picked_cells).reshape(-1,1),  np.tile(max_corrs, (len(st_meta_sc_sub))).reshape(-1,2)])
        picked_cells_arr = np.vstack([picked_cells_arr, st_meta_sc_sub])
    picked_cells_arr_df_without_interface = pd.DataFrame(picked_cells_arr, columns = ['spot', 'cell_ratio', 'celltype', 'x','y','cell_id','cor_exp', 'cor_interface'])
    return picked_cells_arr_df_without_interface