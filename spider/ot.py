import numpy as np
from scipy import sparse
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

# borrowed from COMMOT


def cot_dense(S, D, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, nitermax=1e4, stopthr=1e-8):
    """ Solve the collective optimal transport problem with distance limits.
    
    Parameters
    ----------
    S : (n_pos_s,ns_s) numpy.ndarray
        Source distributions over `n_pos_s` positions of `ns_s` source species.
    D : (n_pos_d,ns_d) numpy.ndarray
        Destination distributions over `n_pos_d` positions of `ns_d` destination species.
    A : (ns_s,ns_d) numpy.ndarray
        The cost coefficients for source-destination species pairs. An infinity value indicates that the two species cannot be coupled.
    M : (n_pos_s,n_pos_d) numpy.ndarray
        The distance (cost) matrix among the positions.
    cutoff : (ns_s,ns_d) numpy.ndarray
        The distance (cost) cutoff between each source-destination species pair. All transports are restricted by the cutoffs.
    eps_p : float, defaults to 1e-1
        The coefficient for entropy regularization of P.
    eps_mu : float, defaults to eps_p
        The coefficient for entropy regularization of unmatched source mass.
    eps_nu : float, defaults to eps_p
        The coefficient for entriopy regularization of unmatched target mass.
    rho : float, defaults to 1e2
        The coefficient for penalizing unmatched mass.
    nitermax : int, optional
        The maximum number of iterations in the unormalized OT problem. Defaults to 1e4.
    stopthr : float, optional
        The relatitive error threshold for terminating the iteration. Defaults to 1e-8.
    
    Returns
    -------
    (ns_s,ns_d,n_pos_s,n_pos_d) numpy.ndarray
        The transport plans among the multiple species.
    """
    np.set_printoptions(precision=2)
    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape
    max_amount = max( S.sum(), D.sum() )
    S = S / max_amount
    D = D / max_amount

    if eps_mu is None: eps_mu = eps_p
    if eps_nu is None: eps_nu = eps_p
    if max(abs(eps_p-eps_mu),abs(eps_p-eps_nu)) > 1e-8:
        unot_solver = "momentum"
    else:
        unot_solver = "sinkhorn"

    # Set up the large collective OT problem
    a = S.flatten('F')
    b = D.flatten('F')
    C = np.inf * np.ones([len(a),len(b)])
    for i in range(ns_s):
        for j in range(ns_d):
            if not np.isinf(A[i,j]):
                tmp_M = np.array(M)
                tmp_M[np.where(tmp_M > cutoff[i,j])] = np.inf
                C[i*n_pos_s:(i+1)*n_pos_s, j*n_pos_d:(j+1)*n_pos_d] = A[i,j] * tmp_M
    C = C/np.max(C[np.where(~np.isinf(C))])
    nzind_a = np.where(a > 0)[0]
    nzind_b = np.where(b > 0)[0]
    tmp_P = unot(a[nzind_a], b[nzind_b], C[nzind_a,:][:,nzind_b], eps_p, rho, \
        eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=False, solver=unot_solver, nitermax=nitermax, stopthr=stopthr)
    P = np.zeros_like(C)
    for i in range(len(nzind_a)):
        for j in range(len(nzind_b)):
            P[nzind_a[i],nzind_b[j]] = tmp_P[i,j]
    P_expand = np.zeros([ns_s, ns_d, n_pos_s, n_pos_d], float)
    for i in range(ns_s):
        for j in range(ns_d):
            P_expand[i,j,:,:] = P[i*n_pos_s:(i+1)*n_pos_s,j*n_pos_d:(j+1)*n_pos_d]
    return P_expand * max_amount

def cot_row_dense(S, D, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, nitermax=1e4, stopthr=1e-8):
    """Solve for each sender species separately.
    """
    if eps_mu is None: eps_mu = eps_p
    if eps_nu is None: eps_nu = eps_p
    if max(abs(eps_p-eps_mu),abs(eps_p-eps_nu)) > 1e-8:
        unot_solver = "momentum"
    else:
        unot_solver = "sinkhorn"

    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape
    P_expand = np.zeros([ns_s, ns_d, n_pos_s, n_pos_d], float)
    for i in range(ns_s):
        a = S[:,i]
        D_ind = np.where(~np.isinf(A[i,:]))[0]
        b = D[:,D_ind].flatten('F')
        max_amount = max(a.sum(), b.sum())
        a = a / max_amount; b = b / max_amount
        C = np.inf * np.ones([len(a), len(b)], float)
        for j in range(len(D_ind)):
            D_j = D_ind[j]
            tmp_M = np.array(M)
            tmp_M[np.where(tmp_M > cutoff[i,D_j])] = np.inf
            C[:,j*n_pos_d:(j+1)*n_pos_d] = A[i,D_j] * tmp_M
        C = C/np.max(C[np.where(~np.isinf(C))])
        nzind_a = np.where(a > 0)[0]
        nzind_b = np.where(b > 0)[0]
        tmp_P = unot(a[nzind_a], b[nzind_b], C[nzind_a,:][:,nzind_b], eps_p, rho, \
            eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=False, solver=unot_solver, nitermax=nitermax, stopthr=stopthr)
        P = np.zeros_like(C)
        for ii in range(len(nzind_a)):
            for jj in range(len(nzind_b)):
                P[nzind_a[ii],nzind_b[jj]] = tmp_P[ii,jj]
        for j in range(len(D_ind)):
            P_expand[i,D_ind[j],:,:] = P[:,j*n_pos_d:(j+1)*n_pos_d] * max_amount
    return P_expand

def cot_col_dense(S, D, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, nitermax=1e4, stopthr=1e-8):
    """Solve for each destination species separately.
    """
    if eps_mu is None: eps_mu = eps_p
    if eps_nu is None: eps_nu = eps_p
    if max(abs(eps_p-eps_mu),abs(eps_p-eps_nu)) > 1e-8:
        unot_solver = "momentum"
    else:
        unot_solver = "sinkhorn"

    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape
    P_expand = np.zeros([ns_s, ns_d, n_pos_s, n_pos_d], float)
    for j in range(ns_d):
        b = D[:,j]
        S_ind = np.where(~np.isinf(A[:,j]))[0]
        a = S[:,S_ind].flatten('F')
        max_amount = max(a.sum(), b.sum())
        a = a / max_amount; b = b / max_amount
        C = np.inf * np.ones([len(a), len(b)], float)
        for i in range(len(S_ind)):
            S_i = S_ind[i]
            tmp_M = np.array(M)
            tmp_M[np.where(tmp_M > cutoff[S_i,j])] = np.inf
            C[i*n_pos_s:(i+1)*n_pos_s,:] = A[S_i,j] * tmp_M
        C = C/np.max(C[np.where(~np.isinf(C))])
        nzind_a = np.where(a > 0)[0]
        nzind_b = np.where(b > 0)[0]
        tmp_P = unot(a[nzind_a], b[nzind_b], C[nzind_a,:][:,nzind_b], eps_p, rho, \
            eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=False, solver=unot_solver, nitermax=nitermax, stopthr=stopthr)
        P = np.zeros_like(C)
        for ii in range(len(nzind_a)):
            for jj in range(len(nzind_b)):
                P[nzind_a[ii],nzind_b[jj]] = tmp_P[ii,jj]
        for i in range(len(S_ind)):
            P_expand[S_ind[i],j,:,:] = P[i*n_pos_s:(i+1)*n_pos_s,:] * max_amount
    return P_expand

def cot_blk_dense(S, D, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, nitermax=1e4, stopthr=1e-8):
    """Solve for each pair of species separately.
    """
    if eps_mu is None: eps_mu = eps_p
    if eps_nu is None: eps_nu = eps_p
    if max(abs(eps_p-eps_mu),abs(eps_p-eps_nu)) > 1e-8:
        unot_solver = "momentum"
    else:
        unot_solver = "sinkhorn"
    
    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape
    P_expand = np.zeros([ns_s, ns_d, n_pos_s, n_pos_d], float)
    for i in range(ns_s):
        for j in range(ns_d):
            if np.isinf(A[i,j]): continue
            a = S[:,i]; b = D[:,j]
            max_amount = max(a.sum(), b.sum())
            a = a / max_amount; b = b / max_amount
            C = np.array(M)
            C[np.where(C > cutoff[i,j])] = np.inf
            C = C/np.max(C[np.where(~np.isinf(C))])
            nzind_a = np.where(a > 0)[0]
            nzind_b = np.where(b > 0)[0]
            tmp_P = unot(a[nzind_a], b[nzind_b], C[nzind_a,:][:,nzind_b], eps_p, rho, \
                eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=False, solver=unot_solver, nitermax=nitermax, stopthr=stopthr)
            P = np.zeros_like(C)
            for ii in range(len(nzind_a)):
                for jj in range(len(nzind_b)):
                    P[nzind_a[ii],nzind_b[jj]] = tmp_P[ii,jj]
            P_expand[i,j,:,:] = P[:,:]
    return P_expand

def run_p(S, D, A, M, cutoff, eps_p, eps_mu, eps_nu, rho, nitermax, stopthr, name):
    if name == 'cot_sparse':
        return cot_sparse(S, D, A, M, cutoff, \
        eps_p=eps_p, eps_mu=eps_mu, eps_nu=eps_nu, rho=rho, \
        nitermax=nitermax, stopthr=stopthr, verbose=False)
    elif name == 'cot_row_sparse':
        return cot_row_sparse(S, D, A, M, cutoff, \
        eps_p=eps_p, eps_mu=eps_mu, eps_nu=eps_nu, rho=rho, \
        nitermax=nitermax, stopthr=stopthr, verbose=False)
    elif name == 'cot_col_sparse':
        return cot_col_sparse(S, D, A, M, cutoff, \
        eps_p=eps_p, eps_mu=eps_mu, eps_nu=eps_nu, rho=rho, \
        nitermax=nitermax, stopthr=stopthr, verbose=False)
    elif name == 'cot_blk_sparse':
        return cot_blk_sparse(S, D, A, M, cutoff, \
        eps_p=eps_p, eps_mu=eps_mu, eps_nu=eps_nu, rho=rho, \
        nitermax=nitermax, stopthr=stopthr, verbose=False)
    else:
        print(f'Unknown method name {name}.')
        

def cot_combine_sparse(S, D, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, weights=(0.1,0.1,0.1,0.7), nitermax=1e4, stopthr=1e-8, verbose=False):
    # from joblib import Parallel, delayed
    # from tqdm import tqdm
    if isinstance(eps_p, tuple):
        eps_p_cot, eps_p_row, eps_p_col, eps_p_blk = eps_p
    else:
        eps_p_cot = eps_p_row = eps_p_col = eps_p_blk = eps_p
    if isinstance(rho, tuple):
        rho_cot, rho_row, rho_col, rho_blk = rho
    else:
        rho_cot = rho_row = rho_col = rho_blk = rho
    if eps_mu is None:
        eps_mu_cot = eps_p_cot; eps_mu_row = eps_p_row
        eps_mu_col = eps_p_col; eps_mu_blk = eps_p_blk
    elif isinstance(eps_mu, tuple):
        eps_mu_cot, eps_mu_row, eps_mu_col, eps_mu_blk = eps_mu
    else:
        eps_mu_cot = eps_mu_row = eps_mu_col = eps_mu_blk = eps_mu
    if eps_nu is None:
        eps_nu_cot = eps_p_cot; eps_nu_row = eps_p_row
        eps_nu_col = eps_p_col; eps_nu_blk = eps_p_blk
    elif isinstance(eps_nu, tuple):
        eps_nu_cot, eps_nu_row, eps_nu_col, eps_nu_blk = eps_nu
    else:
        eps_nu_cot = eps_nu_row = eps_nu_col = eps_nu_blk = eps_nu

    P_cot = cot_sparse(S, D, A, M, cutoff, \
        eps_p=eps_p_cot, eps_mu=eps_mu_cot, eps_nu=eps_nu_cot, rho=rho_cot, \
        nitermax=nitermax, stopthr=stopthr, verbose=False)
    P_row = cot_row_sparse(S, D, A, M, cutoff, \
        eps_p=eps_p_row, eps_mu=eps_mu_row, eps_nu=eps_nu_row, rho=rho_row, \
        nitermax=nitermax, stopthr=stopthr, verbose=False)
    P_col = cot_col_sparse(S, D, A, M, cutoff, \
        eps_p=eps_p_col, eps_mu=eps_mu_col, eps_nu=eps_nu_col, rho=rho_col, \
        nitermax=nitermax, stopthr=stopthr, verbose=False)
    P_blk = cot_blk_sparse(S, D, A, M, cutoff, \
        eps_p=eps_p_blk, eps_mu=eps_mu_blk, eps_nu=eps_nu_blk, rho=rho_blk, \
        nitermax=nitermax, stopthr=stopthr, verbose=False)
    
    # input_arr = [(eps_p_cot, eps_mu_cot, eps_nu_cot, rho_cot, 'cot_sparse'),
    #              (eps_p_row, eps_mu_row, eps_nu_row, rho_row, 'cot_row_sparse'),
    #              (eps_p_col, eps_mu_col, eps_nu_col, rho_col, 'cot_col_sparse'),
    #              (eps_p_blk, eps_mu_blk, eps_nu_blk, rho_blk, 'cot_blk_sparse')]
    

    # [P_cot, P_row, P_col, P_blk] = Parallel(n_jobs=4)(
    #     delayed(run_p)(S, D, A, M, cutoff, input1[0], input1[1], input1[2], input1[3], nitermax, stopthr, input1[4])\
    #         for input1 in tqdm(input_arr, desc="Running COT")
    # )

    P = {}
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if not np.isinf(A[i,j]):
                P[(i,j)] = float(weights[0]) * P_cot[(i,j)] + float(weights[1]) * P_row[(i,j)] \
                    + float(weights[2]) * P_col[(i,j)] + float(weights[3]) * P_blk[(i,j)]
    return(P)

def cot_sparse(S, D, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, nitermax=1e4, stopthr=1e-8, verbose=False):
    """ Solve the collective optimal transport problem with distance limits in sparse format.
    
    Parameters
    ----------
    S : (n_pos_s,ns_s) numpy.ndarray
        Source distributions over `n_pos_s` positions of `ns_s` source species.
    D : (n_pos_d,ns_d) numpy.ndarray
        Destination distributions over `n_pos_d` positions of `ns_d` destination species.
    A : (ns_s,ns_d) numpy.ndarray
        The cost coefficients for source-destination species pairs. An infinity value indicates that the two species cannot be coupled.
    M : (n_pos_s,n_pos_d) numpy.ndarray
        The distance (cost) matrix among the positions.
    cutoff : (ns_s,ns_d) numpy.ndarray
        The distance (cost) cutoff between each source-destination species pair. All transports are restricted by the cutoffs.
    eps_p : float, defaults to 1e-1
        The coefficient for entropy regularization of P.
    eps_mu : float, defaults to eps_p
        The coefficient for entropy regularization of unmatched source mass.
    eps_nu : float, defaults to eps_p
        The coefficient for entriopy regularization of unmatched target mass.
    rho : float, defaults to 1e2
        The coefficient for penalizing unmatched mass.
    nitermax : int, optional
        The maximum number of iterations in the unormalized OT problem. Defaults to 1e4.
    stopthr : float, optional
        The relatitive error threshold for terminating the iteration. Defaults to 1e-8.
    
    Returns
    -------
    A dictionary of scipy.sparse.coo_matrix
        The transport plan in coo sparse format for source species i and destinaton species j can be retrieved with the key (i,j).
    """
    np.set_printoptions(precision=2)
    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape
    max_amount = max( S.sum(), D.sum() )
    S = S / max_amount
    D = D / max_amount

    if eps_mu is None: eps_mu = eps_p
    if eps_nu is None: eps_nu = eps_p
    if max(abs(eps_p-eps_mu),abs(eps_p-eps_nu)) > 1e-8:
        unot_solver = "momentum"
    else:
        unot_solver = "sinkhorn"

    # Set up the large collective OT problem
    a = S.flatten('F')
    b = D.flatten('F')
    
    C_data, C_row, C_col = [], [], []

    max_cutoff = cutoff.max()
    M_row, M_col = np.where(M <= max_cutoff)
    M_max_sp = sparse.coo_matrix((M[M_row,M_col], (M_row,M_col)), shape=M.shape)
    
    cost_scales = []
    for i in range(ns_s):
        for j in range(ns_d):
            if not np.isinf(A[i,j]):
                tmp_nzind_s = np.where(S[:,i] > 0)[0]
                tmp_nzind_d = np.where(D[:,j] > 0)[0]
                tmp_M_max_sp = coo_submatrix_pull(M_max_sp, tmp_nzind_s, tmp_nzind_d)
                tmp_ind = np.where(tmp_M_max_sp.data <= cutoff[i,j])[0]
                tmp_row = tmp_nzind_s[tmp_M_max_sp.row[tmp_ind]]
                tmp_col = tmp_nzind_d[tmp_M_max_sp.col[tmp_ind]]
                cost_scales.append(np.max(M_max_sp.data[np.where(M_max_sp.data <= cutoff[i,j])])*A[i,j])
                C_data.append( tmp_M_max_sp.data[tmp_ind]*A[i,j] )
                C_row.append( tmp_row+i*n_pos_s )
                C_col.append( tmp_col+j*n_pos_d )

    cost_scale = np.max(cost_scales)
    C_data = np.concatenate(C_data, axis=0)
    C_row = np.concatenate(C_row, axis=0)
    C_col = np.concatenate(C_col, axis=0)
    C = sparse.coo_matrix((C_data/cost_scale, (C_row, C_col)), shape=(len(a),len(b)))

    # Solve the problem on nonzero mass
    nzind_a = np.where(a > 0)[0]
    nzind_b = np.where(b > 0)[0]
    C_nz = coo_submatrix_pull(C, nzind_a, nzind_b)

    if verbose:
        print('Number of non-infinity entries in transport cost:', len(C.data))

    del C_data, C_row, C_col, C

    tmp_P = unot(a[nzind_a], b[nzind_b], C_nz, eps_p, rho, \
        eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=True, solver=unot_solver, nitermax=nitermax, stopthr=stopthr)
    
    del C_nz

    P = sparse.coo_matrix((tmp_P.data, (nzind_a[tmp_P.row], nzind_b[tmp_P.col])), shape=(len(a),len(b)))
    P = P.tocsr()

    # Output a dictionary of transport plans
    P_expand = {}
    for i in range(ns_s):
        for j in range(ns_d):
            if not np.isinf(A[i,j]):
                tmp_P = P[i*n_pos_s:(i+1)*n_pos_s, j*n_pos_d:(j+1)*n_pos_d]
                P_expand[(i,j)] = tmp_P.tocoo() * max_amount

    return P_expand    

def cot_row_sparse(S, D, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, nitermax=1e4, stopthr=1e-8, verbose=False):
    """Solve for each sender species separately.
    """
    if eps_mu is None: eps_mu = eps_p
    if eps_nu is None: eps_nu = eps_p
    if max(abs(eps_p-eps_mu),abs(eps_p-eps_nu)) > 1e-8:
        unot_solver = "momentum"
    else:
        unot_solver = "sinkhorn"

    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape

    max_cutoff = cutoff.max()
    M_row, M_col = np.where(M <= max_cutoff)
    M_max_sp = sparse.coo_matrix((M[M_row,M_col], (M_row,M_col)), shape=M.shape)
    
    P_expand = {}
    for i in range(ns_s):
        a = S[:,i]
        D_ind = np.where(~np.isinf(A[i,:]))[0]
        b = D[:,D_ind].flatten('F')
        nzind_a = np.where(a > 0)[0]; nzind_b = np.where(b > 0)[0]
        if len(nzind_a)==0 or len(nzind_b)==0:
            for j in range(len(D_ind)):
                P_expand[(i,D_ind[j])] = sparse.coo_matrix(([],([],[])), shape=(n_pos_s, n_pos_d), dtype=float)
            continue
        max_amount = max(a.sum(), b.sum())
        a = a / max_amount; b = b / max_amount
        C_data, C_row, C_col = [], [], []
        cost_scales = []
        for j in range(len(D_ind)):
            D_j = D_ind[j]
            tmp_nzind_s = np.where(S[:,i] > 0)[0]
            tmp_nzind_d = np.where(D[:,D_j] > 0)[0]
            tmp_M_max_sp = coo_submatrix_pull(M_max_sp, tmp_nzind_s, tmp_nzind_d)
            tmp_ind = np.where(tmp_M_max_sp.data <= cutoff[i,D_j])[0]
            tmp_row = tmp_nzind_s[tmp_M_max_sp.row[tmp_ind]]
            tmp_col = tmp_nzind_d[tmp_M_max_sp.col[tmp_ind]]
            C_data.append( tmp_M_max_sp.data[tmp_ind]*A[i,D_j] )
            C_row.append( tmp_row )
            C_col.append( tmp_col+j*n_pos_d )
            cost_scales.append( np.max(M_max_sp.data[np.where(M_max_sp.data <= cutoff[i,D_j])])*A[i,D_j] )
        cost_scale = np.max(cost_scales)
        C_data = np.concatenate(C_data, axis=0)
        C_row = np.concatenate(C_row, axis=0)
        C_col = np.concatenate(C_col, axis=0)
        C = sparse.coo_matrix((C_data/cost_scale, (C_row, C_col)), shape=(len(a), len(b)))    

        nzind_a = np.where(a > 0)[0]
        nzind_b = np.where(b > 0)[0]
        C_nz = coo_submatrix_pull(C, nzind_a, nzind_b)

        del C_data, C_row, C_col, C

        tmp_P = unot(a[nzind_a], b[nzind_b], C_nz, eps_p, rho, \
            eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=True, solver=unot_solver, nitermax=nitermax, stopthr=stopthr)

        del C_nz

        P = sparse.coo_matrix((tmp_P.data, (nzind_a[tmp_P.row], nzind_b[tmp_P.col])), shape=(len(a),len(b)))
        P = P.tocsr()

        for j in range(len(D_ind)):
            tmp_P = P[:,j*n_pos_d:(j+1)*n_pos_d]
            P_expand[(i,D_ind[j])] = tmp_P.tocoo() * max_amount

        del P

    return P_expand

def cot_col_sparse(S, D, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, nitermax=1e4, stopthr=1e-8, verbose=False):
    """Solve for each destination species separately.
    """
    if eps_mu is None: eps_mu = eps_p
    if eps_nu is None: eps_nu = eps_p
    if max(abs(eps_p-eps_mu),abs(eps_p-eps_nu)) > 1e-8:
        unot_solver = "momentum"
    else:
        unot_solver = "sinkhorn"

    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape

    max_cutoff = cutoff.max()
    M_row, M_col = np.where(M <= max_cutoff)
    M_max_sp = sparse.coo_matrix((M[M_row,M_col], (M_row,M_col)), shape=M.shape)
    
    P_expand = {}
    for j in range(ns_d):
        S_ind = np.where(~np.isinf(A[:,j]))[0]
        a = S[:,S_ind].flatten('F')
        b = D[:,j]
        nzind_a = np.where(a > 0)[0]; nzind_b = np.where(b > 0)[0]
        if len(nzind_a)==0 or len(nzind_b)==0:
            for i in range(len(S_ind)):
                P_expand[(S_ind[i],j)] = sparse.coo_matrix(([],([],[])), shape=(n_pos_s,n_pos_d), dtype=float)
            continue
        max_amount = max(a.sum(), b.sum())
        a = a / max_amount; b = b / max_amount
        C_data, C_row, C_col = [], [], []
        cost_scales = []
        for i in range(len(S_ind)):
            S_i = S_ind[i]
            tmp_nzind_s = np.where(S[:,S_i] > 0)[0]
            tmp_nzind_d = np.where(D[:,j] > 0)[0]
            tmp_M_max_sp = coo_submatrix_pull(M_max_sp, tmp_nzind_s, tmp_nzind_d)
            tmp_ind = np.where(tmp_M_max_sp.data <= cutoff[S_i,j])[0]
            tmp_row = tmp_nzind_s[tmp_M_max_sp.row[tmp_ind]]
            tmp_col = tmp_nzind_d[tmp_M_max_sp.col[tmp_ind]]
            C_data.append( tmp_M_max_sp.data[tmp_ind]*A[S_i,j] )
            C_row.append( tmp_row+i*n_pos_s )
            C_col.append( tmp_col )
            cost_scales.append( np.max(M_max_sp.data[np.where(M_max_sp.data <= cutoff[S_i,j])])*A[S_i,j] )
        cost_scale = np.max(cost_scales)
        C_data = np.concatenate(C_data, axis=0)
        C_row = np.concatenate(C_row, axis=0)
        C_col = np.concatenate(C_col, axis=0)
        C = sparse.coo_matrix((C_data/cost_scale, (C_row, C_col)), shape=(len(a), len(b)))    

        nzind_a = np.where(a > 0)[0]
        nzind_b = np.where(b > 0)[0]
        C_nz = coo_submatrix_pull(C, nzind_a, nzind_b)

        del C_data, C_row, C_col, C

        tmp_P = unot(a[nzind_a], b[nzind_b], C_nz, eps_p, rho, \
            eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=True, solver=unot_solver, nitermax=nitermax, stopthr=stopthr)

        del C_nz

        P = sparse.coo_matrix((tmp_P.data, (nzind_a[tmp_P.row], nzind_b[tmp_P.col])), shape=(len(a),len(b)))
        P = P.tocsr()

        for i in range(len(S_ind)):
            tmp_P = P[i*n_pos_s:(i+1)*n_pos_s,:]
            P_expand[(S_ind[i],j)] = tmp_P.tocoo() * max_amount

        del P

    return P_expand

def cot_blk_sparse(S, D, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, nitermax=1e4, stopthr=1e-8, verbose=False):
    if eps_mu is None: eps_mu = eps_p
    if eps_nu is None: eps_nu = eps_p
    if max(abs(eps_p-eps_mu), abs(eps_p-eps_nu)) > 1e-8:
        unot_solver = "momentum"
    else:
        unot_solver = "sinkhorn"
    
    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape

    max_cutoff = cutoff.max()
    M_row, M_col = np.where(M <= max_cutoff)
    M_max_sp = sparse.coo_matrix((M[M_row,M_col], (M_row,M_col)), shape=M.shape)

    P_expand = {}
    for i in range(ns_s):
        for j in range(ns_d):
            if not np.isinf(A[i,j]):
                a = S[:,i]; b = D[:,j]
                nzind_a = np.where(a > 0)[0]; nzind_b = np.where(b > 0)[0]
                if len(nzind_a)==0 or len(nzind_b)==0:
                    P_expand[(i,j)] = sparse.coo_matrix(([],([],[])), shape=(n_pos_s, n_pos_d), dtype=float)
                    continue
                max_amount = max(a.sum(), b.sum())
                a = a / max_amount; b = b / max_amount
                tmp_nzind_s = np.where(S[:,i] > 0)[0]
                tmp_nzind_d = np.where(D[:,j] > 0)[0]
                tmp_M_max_sp = coo_submatrix_pull(M_max_sp, tmp_nzind_s, tmp_nzind_d)
                tmp_ind = np.where(tmp_M_max_sp.data <= cutoff[i,j])[0]
                tmp_row = tmp_nzind_s[tmp_M_max_sp.row[tmp_ind]]
                tmp_col = tmp_nzind_d[tmp_M_max_sp.col[tmp_ind]]
                C_data = tmp_M_max_sp.data[tmp_ind] * A[i,j]
                cost_scale = np.max( M_max_sp.data[np.where(M_max_sp.data <= cutoff[i,j])] )*A[i,j]
                C = sparse.coo_matrix((C_data/cost_scale, (tmp_row, tmp_col)), shape=(len(a), len(b)))

                nzind_a = np.where(a > 0)[0]
                nzind_b = np.where(b > 0)[0]
                C_nz = coo_submatrix_pull(C, nzind_a, nzind_b)

                del C_data, C

                tmp_P = unot(a[nzind_a], b[nzind_b], C_nz, eps_p, rho, \
                    eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=True, solver=unot_solver, nitermax=nitermax, stopthr=stopthr)

                del C_nz

                P = sparse.coo_matrix((tmp_P.data, (nzind_a[tmp_P.row], nzind_b[tmp_P.col])), shape=(len(a),len(b)))

                P_expand[(i,j)] = P * max_amount
    
    return P_expand

def coo_submatrix_pull(matr, rows, cols):
    """
    Pulls out an arbitrary i.e. non-contiguous submatrix out of
    a sparse.coo_matrix. 
    """
    if type(matr) != sparse.coo_matrix:
        raise TypeError('Matrix must be sparse COOrdinate format')
    
    gr = -1 * np.ones(matr.shape[0])
    gc = -1 * np.ones(matr.shape[1])
    
    lr = len(rows)
    lc = len(cols)
    
    ar = np.arange(0, lr)
    ac = np.arange(0, lc)
    gr[rows[ar]] = ar
    gc[cols[ac]] = ac
    mrow = matr.row
    mcol = matr.col
    newelem = (gr[mrow] > -1) & (gc[mcol] > -1)
    newrows = mrow[newelem]
    newcols = mcol[newelem]
    return sparse.coo_matrix((matr.data[newelem], np.array([gr[newrows],
        gc[newcols]])),(lr, lc))

def unot(a,
        b,
        C,
        eps_p,
        rho,
        eps_mu=None,
        eps_nu=None,
        sparse_mtx=False,
        solver="sinkhorn",
        nitermax=10000,
        stopthr=1e-8,
        verbose=False,
        momentum_dt=0.1,
        momentum_beta=0.0):
    """ The main function calling different algorithms.

    Parameters
    ----------
    a : (ns,) numpy.ndarray
        Source distribution. The summation should be less than or equal to 1.
    b : (nt,) numpy.ndarray
        Target distribution. The summation should be less than or equal to 1.
    C : (ns,nt) numpy.ndarray
        The cost matrix possibly with infinity entries.
    eps_p :  float
        The coefficient of entropy regularization for P.
    rho : float
        The coefficient of penalty for unmatched mass.
    eps_mu : float, defaults to eps_p
        The coefficient of entropy regularization for mu.
    eps_nu : float, defaults to eps_p
        The coefficient of entropy regularization for nu.
    sparse_mtx : boolean, defaults to False
        Whether using sparse matrix format. If True, C should be in coo_sparse format.
    solver : str, defaults to 'sinkhorn'
        The solver to use. Choose from 'sinkhorn' and 'momentum'.
    nitermax : int, defaults to 10000
        The maximum number of iterations.
    stopthr : float, defaults to 1e-8
        The relative error threshold for stopping.
    verbose : boolean, defaults to False
        Whether to print algorithm logs.
    momentum_dt : float, defaults to 1e-1
        Step size if momentum method is used.
    momentum_beta : float, defautls to 0
        The coefficient for the momentum term if momemtum method is used.
    """
    if eps_mu is None: eps_mu = eps_p
    if eps_nu is None: eps_nu = eps_p
    # Return a zero matrix if either a or b is all zero
    nzind_a = np.where(a > 0)[0]; nzind_b = np.where(b > 0)[0]
    if len(nzind_a) == 0 or len(nzind_b) == 0:
        if sparse_mtx:
            P = sparse.coo_matrix(([],([],[])), shape=(len(a), len(b)))
        else:
            P = np.zeros([len(a), len(b)], float)
        return P
    if solver == "sinkhorn" and max(abs(eps_p-eps_mu),abs(eps_p-eps_nu))>1e-8:
        print("To use Sinkhorn algorithm, set eps_p=eps_mu=eps_nu")
        exit()
    if solver == "sinkhorn" and not sparse_mtx:
        P = unot_sinkhorn_l1_dense(a,b,C,eps_p,rho, \
            nitermax=nitermax,stopthr=stopthr,verbose=verbose)
    elif solver == "sinkhorn" and sparse_mtx:
        P = unot_sinkhorn_l1_sparse(a,b,C,eps_p,rho, \
            nitermax=nitermax,stopthr=stopthr,verbose=verbose)
    elif solver == "momentum" and not sparse_mtx: 
        P = unot_momentum_l1_dense(a,b,C,eps_p,eps_mu,eps_nu,rho, \
            nitermax=nitermax,stopthr=stopthr,dt=momentum_dt, \
            beta=momentum_beta,precondition=True,verbose=verbose)
    elif solver == "momentum" and sparse_mtx:
        print("under construction")
        exit()
    return P

def unot_sinkhorn_l2_dense(a,b,C,eps,m,nitermax=10000,stopthr=1e-8,verbose=False):
    """ Solve the unnormalized optimal transport with l2 penalty in dense matrix format.

    Parameters
    ----------
    a : (ns,) numpy.ndarray
        Source distribution. The summation should be less than or equal to 1.
    b : (nt,) numpy.ndarray
        Target distribution. The summation should be less than or equal to 1.
    C : (ns,nt) numpy.ndarray
        The cost matrix possibly with infinity entries.
    eps : float
        The coefficient for entropy regularization.
    m : float
        The coefficient for penalizing unmatched mass.
    nitermax : int, optional
        The max number of iterations. Defaults to 10000.
    stopthr : float, optional
        The threshold for terminating the iteration. Defaults to 1e-8.

    Returns
    -------
    (ns,nt) numpy.ndarray
        The optimal transport matrix.
    """
    f = np.zeros_like(a)
    g = np.zeros_like(b)
    r = np.zeros_like(a)
    s = np.zeros_like(b)
    niter = 0
    err = 100
    while niter <= nitermax and err > stopthr:
        fprev = f
        gprev = g
        # Iteration
        f = eps * np.log(a) \
            - eps * np.log( np.sum( np.exp( ( f.reshape(-1,1) + g.reshape(1,-1) - C ) / eps ), axis=1 ) \
            + np.exp( ( r + f ) / eps ) ) + f
        g = eps * np.log(b) \
            - eps * np.log( np.sum( np.exp( ( f.reshape(-1,1) + g.reshape(1,-1) - C ) / eps ), axis=0 ) \
            + np.exp( ( s + g ) / eps ) ) + g
        r = - eps * wrightomega( f/eps - np.log( eps/m ) ).real
        s = - eps * wrightomega( g/eps - np.log( eps/m ) ).real
        # Check relative error
        if niter % 10 == 0:
            err_f = abs(f - fprev).max() / max(abs(f).max(), abs(fprev).max(), 1.)
            err_g = abs(g - gprev).max() / max(abs(g).max(), abs(gprev).max(), 1.)
            err = 0.5 * (err_f + err_g)
        niter = niter + 1
    if verbose:
        print('Number of iterations in unot:', niter)
    P = np.exp( ( f.reshape(-1,1) + g.reshape(1,-1) - C ) / eps )
    return P

def unot_sinkhorn_l1_dense(a,b,C,eps,m,nitermax=10000,stopthr=1e-8,verbose=False,output_fg=False):
    """ Solve the unnormalized optimal transport with l1 penalty in dense matrix format.

    Parameters
    ----------
    a : (ns,) numpy.ndarray
        Source distribution. The summation should be less than or equal to 1.
    b : (nt,) numpy.ndarray
        Target distribution. The summation should be less than or equal to 1.
    C : (ns,nt) numpy.ndarray
        The cost matrix possibly with infinity entries.
    eps : float
        The coefficient for entropy regularization.
    m : float
        The coefficient for penalizing unmatched mass.
    nitermax : int, optional
        The max number of iterations. Defaults to 10000.
    stopthr : float, optional
        The threshold for terminating the iteration. Defaults to 1e-8.

    Returns
    -------
    (ns,nt) numpy.ndarray
        The optimal transport matrix.
    """
    f = np.zeros_like(a)
    g = np.zeros_like(b)
    niter = 0
    err = 100
    while niter <= nitermax and err >  stopthr:
        fprev = f
        gprev = g
        # Iteration
        f = eps * np.log(a) \
            - eps * np.log( np.sum( np.exp( ( f.reshape(-1,1) + g.reshape(1,-1) - C ) / eps ), axis=1 ) \
            + np.exp( ( -m + f ) / eps ) ) + f
        g = eps * np.log(b) \
            - eps * np.log( np.sum( np.exp( ( f.reshape(-1,1) + g.reshape(1,-1) - C ) / eps ), axis=0 ) \
            + np.exp( ( -m + g ) / eps ) ) + g
        # Check relative error
        if niter % 10 == 0:
            err_f = abs(f - fprev).max() / max(abs(f).max(), abs(fprev).max(), 1.)
            err_g = abs(g - gprev).max() / max(abs(g).max(), abs(gprev).max(), 1.)
            err = 0.5 * (err_f + err_g)
        niter = niter + 1
    if verbose:
        print('Number of iterations in unot:', niter)
    P = np.exp( ( f.reshape(-1,1) + g.reshape(1,-1) - C ) / eps )
    if output_fg:
        return f,g
    else:
        return P

def unot_barycenter_sinkhorn_l1_dense(a,C,eps,m,w,nitermax=5):
    L = len(a)
    n = len(a[0])
    f,g = {},{}
    for k in range(L):
        f[k] = np.zeros([n],float)
        g[k] = np.zeros([n],float)

    for i in range(nitermax):
        for k in range(L):
            f[k] = eps * np.log(a[k]) \
                - eps * np.log( np.sum( np.exp( ( f[k].reshape(-1,1) + g[k].reshape(1,-1) - C ) / eps ), axis=1 ) #) + f[k] #\
                + np.exp( ( -m + f[k] ) / eps ) ) + f[k]
        Lnu = np.zeros([n],float)
        for k in range(L):
            Lnu = Lnu + w[k] * np.log( np.sum( np.exp( ( f[k].reshape(-1,1) + g[k].reshape(1,-1) - C ) / eps ), axis=0 ))
        for k in range(L):
            g[k] = eps * Lnu \
                - eps * np.log( np.sum( np.exp( ( f[k].reshape(-1,1) + g[k].reshape(1,-1) - C ) / eps ), axis=0 ) #) + g[k] #\
                + np.exp( ( -m + g[k] ) / eps ) ) + g[k]
    return(np.exp(Lnu))

def regular_barycenter(a,C,eps,w,nitermax=10000):
    L = len(a)
    n = len(a[0])
    f,g = {},{}
    for k in range(L):
        f[k] = np.zeros([n],float)
        g[k] = np.zeros([n],float)

    for i in range(nitermax):
        for k in range(L):
            f[k] = eps * np.log(a[k]) \
                - eps * np.log( np.sum( np.exp( ( f[k].reshape(-1,1) + g[k].reshape(1,-1) - C ) / eps ), axis=1 ) ) + f[k]
        Lnu = np.zeros([n],float)
        for k in range(L):
            Lnu = Lnu + w[k] * np.log( np.sum( np.exp( ( f[k].reshape(-1,1) + g[k].reshape(1,-1) - C ) / eps ), axis=0 ))
        for k in range(L):
            g[k] = eps * Lnu \
                - eps * np.log( np.sum( np.exp( ( f[k].reshape(-1,1) + g[k].reshape(1,-1) - C ) / eps ), axis=0 ) ) + g[k]
    return(np.exp(Lnu))

def unot_sinkhorn_l2_sparse(a,b,C,eps,m,nitermax=10000,stopthr=1e-8,verbose=False):
    """ Solve the unnormalized optimal transport with l2 penalty in sparse matrix format.

    Parameters
    ----------
    a : (ns,) numpy.ndarray
        Source distribution. The summation should be less than or equal to 1.
    b : (nt,) numpy.ndarray
        Target distribution. The summation should be less than or equal to 1.
    C : (ns,nt) scipy.sparse.coo_matrix
        The cost matrix in coo sparse format. The entries exceeds the cost cutoff are omitted. The naturally zero entries should be explicitely included.
    eps : float
        The coefficient for entropy regularization.
    m : float
        The coefficient for penalizing unmatched mass.
    nitermax : int, optional
        The max number of iterations. Defaults to 10000.
    stopthr : float, optional
        The threshold for terminating the iteration. Defaults to 1e-8.

    Returns
    -------
    (ns,nt) scipy.sparse.coo_matrix
        The optimal transport matrix. The locations of entries should agree with C and there might by explicit zero entries.
    """
    tmp_K = C.copy()
    f = np.zeros_like(a)
    g = np.zeros_like(b)
    r = np.zeros_like(a)
    s = np.zeros_like(b)
    niter = 0
    err = 100
    while niter <= nitermax and err > stopthr:
        fprev = f
        gprev = g
        # Iteration
        tmp_K.data = np.exp( ( -C.data + f[C.row] + g[C.col] ) / eps )
        f = eps * np.log(a) \
            - eps * np.log( np.sum( tmp_K, axis=1 ).A.reshape(-1) \
            + np.exp( ( r + f ) / eps ) ) + f
        tmp_K.data = np.exp( ( -C.data + f[C.row] + g[C.col] ) / eps )
        g = eps * np.log(b) \
            - eps * np.log( np.sum( tmp_K, axis=0 ).A.reshape(-1) \
            + np.exp( ( s + g ) / eps ) ) + g
        r = - eps * wrightomega( f/eps - np.log( eps/m ) ).real
        s = - eps * wrightomega( g/eps - np.log( eps/m ) ).real
        # Check relative error
        if niter % 10 == 0:
            err_f = abs(f - fprev).max() / max(abs(f).max(), abs(fprev).max(), 1.)
            err_g = abs(g - gprev).max() / max(abs(g).max(), abs(gprev).max(), 1.)
            err = 0.5 * (err_f + err_g)
        niter = niter + 1
    if verbose:
        print('Number of iterations in unot:', niter)
    tmp_K.data = np.exp( ( -C.data + f[C.row] + g[C.col] ) / eps )
    return tmp_K

def unot_sinkhorn_l1_sparse(a,b,C,eps,m,nitermax=10000,stopthr=1e-8,verbose=False):
    """ Solve the unnormalized optimal transport with l1 penalty in sparse matrix format.

    Parameters
    ----------
    a : (ns,) numpy.ndarray
        Source distribution. The summation should be less than or equal to 1.
    b : (nt,) numpy.ndarray
        Target distribution. The summation should be less than or equal to 1.
    C : (ns,nt) scipy.sparse.coo_matrix
        The cost matrix in coo sparse format. The entries exceeds the cost cutoff are omitted. The naturally zero entries should be explicitely included.
    eps : float
        The coefficient for entropy regularization.
    m : float
        The coefficient for penalizing unmatched mass.
    nitermax : int, optional
        The max number of iterations. Defaults to 10000.
    stopthr : float, optional
        The threshold for terminating the iteration. Defaults to 1e-8.

    Returns
    -------
    (ns,nt) scipy.sparse.coo_matrix
        The optimal transport matrix. The locations of entries should agree with C and there might by explicit zero entries.
    """
    tmp_K = C.copy()
    f = np.zeros_like(a)
    g = np.zeros_like(b)
    r = np.zeros_like(a)
    s = np.zeros_like(b)
    niter = 0
    err = 100
    while niter <= nitermax and err > stopthr:
        fprev = f
        gprev = g
        # Iteration
        tmp_K.data = np.exp( ( -C.data + f[C.row] + g[C.col] ) / eps )
        f = eps * np.log(a) \
            - eps * np.log( np.sum( tmp_K, axis=1 ).A.reshape(-1) \
            + np.exp( ( -m + f ) / eps ) ) + f
        tmp_K.data = np.exp( ( -C.data + f[C.row] + g[C.col] ) / eps )
        g = eps * np.log(b) \
            - eps * np.log( np.sum( tmp_K, axis=0 ).A.reshape(-1) \
            + np.exp( ( -m + g ) / eps ) ) + g
        # Check relative error
        if niter % 10 == 0:
            err_f = abs(f - fprev).max() / max(abs(f).max(), abs(fprev).max(), 1.)
            err_g = abs(g - gprev).max() / max(abs(g).max(), abs(gprev).max(), 1.)
            err = 0.5 * (err_f + err_g)
        niter = niter + 1

    if verbose:
        print('Number of iterations in unot:', niter)
    tmp_K.data = np.exp( ( -C.data + f[C.row] + g[C.col] ) / eps )
    return tmp_K

def unot_nesterov_l2_dense(a,b,C,eps1,eps2,m,nitermax=10000,stopthr=1e-8):
    dt = 0.01
    f = np.zeros_like(a)
    g = np.zeros_like(b)
    f_old = np.array(f)
    g_old = np.array(g)
    F_old = np.array(f)
    G_old = np.array(g)
    niter = 0
    err = 100
    def Qf(ff,gg,ee1,ee2,mm,aa,bb,CC):
        out = np.exp(ff/ee1) * np.sum(np.exp((gg.reshape(1,-1)-CC)/ee1), axis=1) \
            + ee2 * wrightomega( ff*mm/ee2 - np.log( ee2/mm ) ).real / mm - aa
        return out
    def Qg(ff,gg,ee1,ee2,mm,aa,bb,CC):
        out = np.exp(gg/ee1) * np.sum(np.exp((ff.reshape(-1,1)-CC)/ee1), axis=0) \
            + ee2 * wrightomega( gg*mm/ee2 - np.log( ee2/mm ) ).real / mm - bb
        return out
    while niter <= nitermax and err > stopthr:
        f = F_old - dt * Qf(F_old, G_old, eps1, eps2, m, a, b, C)
        F = f + float(niter)/(niter+3.0) * (f-f_old)
        g = G_old - dt * Qg(F_old, G_old, eps1, eps2, m, a, b, C)
        G = g + float(niter)/(niter+3.0) * (g-g_old)
        if niter % 10 == 0:
            err_f = abs(f - f_old).max() / max(abs(f).max(), abs(f_old).max(), 1.)
            err_g = abs(g - g_old).max() / max(abs(g).max(), abs(g_old).max(), 1.)
            err = 0.5 * (err_f + err_g)
        f_old[:] = f[:]; F_old[:] = F[:]
        g_old[:] = g[:]; G_old[:] = G[:]
        niter += 1
    P = np.exp((f.reshape(-1,1)+g.reshape(1,-1)-C)/eps1)
    return P

def unot_momentum_l2_dense(a,b,C,eps_p,eps_mu,eps_nu,m,nitermax=1e4,stopthr=1e-8,dt=0.01,beta=0.8):
    f = np.zeros_like(a)
    g = np.zeros_like(b)
    f_old = np.array(f)
    g_old = np.array(g)
    F_old = np.array(f)
    G_old = np.array(g)
    niter = 0
    err = 100
    def Qf(ff,gg,ee_p,ee_mu,ee_nu,mm,aa,bb,CC):
        out = np.exp(ff/ee_p) * np.sum(np.exp((gg.reshape(1,-1)-CC)/ee_p), axis=1) \
            + ee_mu * wrightomega( ff/ee_mu - np.log( ee_mu/mm ) ).real / mm - aa
        return out
    def Qg(ff,gg,ee_p,ee_mu,ee_nu,mm,aa,bb,CC):
        out = np.exp(gg/ee_p) * np.sum(np.exp((ff.reshape(-1,1)-CC)/ee_p), axis=0) \
            + ee_nu * wrightomega( gg/ee_nu - np.log( ee_nu/mm ) ).real / mm - bb
        return out
    while niter <= nitermax and err > stopthr:
        F = beta * F_old + Qf(f_old, g_old, eps_p, eps_mu, eps_nu, m, a, b, C)
        f = f_old - dt * F
        G = beta * G_old + Qg(f_old, g_old, eps_p, eps_mu, eps_nu, m, a, b, C)
        g = g_old - dt * G
        f_old[:] = f[:]; F_old[:] = F[:]
        g_old[:] = g[:]; G_old[:] = G[:]
        niter += 1
    P = np.exp((f.reshape(-1,1)+g.reshape(1,-1)-C)/eps_p)
    return P

def unot_momentum_l1_dense(a,b,C,eps_p,eps_mu,eps_nu,m,nitermax=1e4,stopthr=1e-8,dt=0.01,beta=0.8,precondition=False,verbose=False):
    f = np.zeros_like(a)
    g = np.zeros_like(b)
    if precondition:
        f,g = unot_sinkhorn_l1_dense(a,b,C,eps_p,m,output_fg=True)
    f_old = np.array(f)
    g_old = np.array(g)
    F_old = np.array(f)
    G_old = np.array(g)
    niter = 0
    err = 100
    def Qf(ff,gg,ee_p,ee_mu,ee_nu,mm,aa,bb,CC):
        out = np.exp(ff/ee_p) * np.sum(np.exp((gg.reshape(1,-1)-CC)/ee_p), axis=1) \
            + np.exp((ff-mm)/ee_mu) - aa
        return out
    def Qg(ff,gg,ee_p,ee_mu,ee_nu,mm,aa,bb,CC):
        out = np.exp(gg/ee_p) * np.sum(np.exp((ff.reshape(-1,1)-CC)/ee_p), axis=0) \
            + np.exp((gg-mm)/ee_nu) - bb
        return out
    while niter <= nitermax and err > stopthr:
        F = beta * F_old + Qf(f_old, g_old, eps_p, eps_mu, eps_nu, m, a, b, C)
        f = f_old - dt * F
        G = beta * G_old + Qg(f_old, g_old, eps_p, eps_mu, eps_nu, m, a, b, C)
        g = g_old - dt * G
        if niter % 10 == 0:
            err_f = abs(f - f_old).max() / max(abs(f).max(), abs(f_old).max(), 1.)
            err_g = abs(g - g_old).max() / max(abs(g).max(), abs(g_old).max(), 1.)
            err = 0.5 * (err_f + err_g)
        f_old[:] = f[:]; F_old[:] = F[:]
        g_old[:] = g[:]; G_old[:] = G[:]
        niter += 1
    P = np.exp((f.reshape(-1,1)+g.reshape(1,-1)-C)/eps_p)
    if verbose:
        print(niter)
    return P

def unot_momentum_l1_2end_dense(a,b,C,eps_p,eps_mu,eps_nu,m,nitermax=1e4,stopthr=1e-8,dt=0.01,beta=0.8):
    f = np.zeros_like(a)
    g = np.zeros_like(b)
    f_old = np.array(f)
    g_old = np.array(g)
    F_old = np.array(f)
    G_old = np.array(g)
    niter = 0
    err = 100
    def Qf(ff,gg,ee_p,ee_mu,ee_nu,mm,aa,bb,CC):
        out = np.exp(ff/ee_p) * np.sum(np.exp((gg.reshape(1,-1)-CC)/ee_p), axis=1) \
            + aa / (np.exp(-(ff-mm)/ee_mu)+1) - aa
        return out
    def Qg(ff,gg,ee_p,ee_mu,ee_nu,mm,aa,bb,CC):
        out = np.exp(gg/ee_p) * np.sum(np.exp((ff.reshape(-1,1)-CC)/ee_p), axis=0) \
            + bb / (np.exp(-(gg-mm)/ee_nu)+1) - bb
        return out
    while niter <= nitermax and err > stopthr:
        F = beta * F_old + Qf(f_old, g_old, eps_p, eps_mu, eps_nu, m, a, b, C)
        f = f_old - dt * F
        G = beta * G_old + Qg(f_old, g_old, eps_p, eps_mu, eps_nu, m, a, b, C)
        g = g_old - dt * G
        f_old[:] = f[:]; F_old[:] = F[:]
        g_old[:] = g[:]; G_old[:] = G[:]
        niter += 1
    P = np.exp((f.reshape(-1,1)+g.reshape(1,-1)-C)/eps_p)
    print(niter)
    return P