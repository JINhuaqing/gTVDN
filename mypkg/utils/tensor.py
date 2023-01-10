#!pip install git+https://github.com/ahwillia/tensortools
import tensorly as tl
import tensortools as tt
from tensortools.operations import unfold as tt_unfold, khatri_rao
import numpy as np

# CPD decomposition when letting first two dim orthogonal for 3-d tensor
def decompose_three_way_orth(tensor, rank, max_iter=501, verbose=False, init=None, eps=1e-3, 
                             is_fixU=False, is_fixV=False):
    
    assert not (is_fixU and (init is None))
    assert not (is_fixV and (init is None))
    
    def _err_fn(vl, v):
        err = np.linalg.norm(vl-v)/np.linalg.norm(v)
        return err

    if init is None:
        aT, _ = np.linalg.qr(np.random.random((rank, tensor.shape[0])).T)
        a = aT.T
        bT, _ = np.linalg.qr(np.random.random((rank, tensor.shape[1])).T)
        b = bT.T
        #c = np.random.random((rank, tensor.shape[2]))
    else:
        aT, bT = init
        a, b = aT.T, bT.T

    last_est = [0, 0, 0]
    for epoch in range(max_iter):
        # optimize c
        input_c = khatri_rao([a.T, b.T])
        target_c = tl.unfold(tensor, mode=2).T
        c = np.linalg.solve(input_c.T.dot(input_c), input_c.T.dot(target_c))
        
        if not is_fixU:
            # optimize a
            input_a = khatri_rao([b.T, c.T])
            target_a = tl.unfold(tensor, mode=0).T
            a = np.linalg.solve(input_a.T.dot(input_a), input_a.T.dot(target_a))
            aT, _ = np.linalg.qr(a.T)
            a = aT.T
            #a = orth(a.T).T

        if not is_fixV:
            # optimize b
            input_b = khatri_rao([a.T, c.T])
            target_b = tl.unfold(tensor, mode=1).T
            b = np.linalg.solve(input_b.T.dot(input_b), input_b.T.dot(target_b))
            bT, _ = np.linalg.qr(b.T)
            b = bT.T
            #b = orth(b.T).T
        
        # calculate error
        al, bl, cl = last_est
        errs = [_err_fn(al, a), _err_fn(bl, b), _err_fn(cl, c)]
        last_est = [a, b, c]
        
        if np.max(errs) < eps:
            break


        if verbose and epoch % int(max_iter * .01) == 0:
            if is_fixU:
                res_a = -1
            else:
                res_a = np.square(input_a.dot(a) - target_a).mean()
            if is_fixV:
                res_b = -1
            else:
                res_b = np.square(input_b.dot(b) - target_b).mean()
            res_c = np.square(input_c.dot(c) - target_c).mean()
            print(f"Epoch: {epoch}, Loss ({res_a:.3f}, {res_b:.3f}, {res_c:.3f}), Err ({errs[0]:.3e}, {errs[1]:.3e}, {errs[2]:.3e}).")

    return a.T, b.T, c.T

# CPD decomposition for 3-d tensor when fixing first two dim
def decompose_three_way_fix(tensor, init=None):

    aT, bT = init
    a, b = aT.T, bT.T

    input_c = khatri_rao([a.T, b.T])
    target_c = tl.unfold(tensor, mode=2).T
    c = np.linalg.solve(input_c.T.dot(input_c), input_c.T.dot(target_c))
        
    return a.T, b.T, c.T


def sort_orthCPD(res):
    """
    args:
        res: a, b, c
    return:
        For the 3-dim tensor,  sort the vector in the last mode via its L2 norm.
    """
    B1, B2, B3 = res
    ws = np.sum(B3**2, axis=0)
    sortIds = np.argsort(-ws)
    sortB1 = B1[:, sortIds]
    sortB2 = B2[:, sortIds]
    sortB3 = B3[:, sortIds]
    return sortB1, sortB2, sortB3


# below, I dont know them exactly
def idenCPD(res):
    """
    args:
        res: the results from parafac function with normalize_factors=True
    return:
        For a D-dim tensor, normalize the first (D-1) components with L2 norm 1  and
                            sort the vector in the last model via its L2 norm.
    """
    factors = res.factors
    weights = res.weights
    sortIdxs = np.argsort(-weights)
    weightsSorted = weights[sortIdxs]
    factorsSorted = [factor[:, sortIdxs] for factor in factors]
    res.weights = np.ones_like(weightsSorted)
    res.factors = factorsSorted
    res.factors[-1] = factorsSorted[-1] * weightsSorted
    return res