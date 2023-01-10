import numpy as np


# eig decomp with sorted results by the mode of eigvals
def eig_sorted(mat):
    eigVals, eigVecs = np.linalg.eig(mat)
        # sort the eigvs and eigvecs
    sidx = np.argsort(-np.abs(eigVals))
    eigVals = eigVals[sidx]
    eigVecs = eigVecs[:, sidx]
    return eigVals, eigVecs

def sorted_jd_result(V, Ds):
    """To fix the order and sign of the results from joint_diag_r
        The order is descending order of L2 norm of the diag terms of D
        The sign is to make most of terms in each col vec be positive
    """
    # fix order of cols of V
    reord_idxs = np.argsort(-np.linalg.norm([np.diag(D) for D in Ds], axis=0))
    reord_Ds = Ds[:, reord_idxs, :][:, :, reord_idxs]
    reord_V = V[:, reord_idxs]
    
    # fix the sign of cols of V
    m, _ = reord_V.shape
    chg_signs = np.ones(m)
    chg_signs[np.mean(reord_V >=0, axis=0) < 0.5] = -1
    reord_V = reord_V*  chg_signs
    
    return reord_V, reord_Ds


def joint_diag_r(As, threshold=1e-8):
    """Python version of function from Sanjay. 
        args:
            As: n x m x m array, n real matrices
            threshold: stopping threshold
        return:
            V: is a m by m orthogonal matrix.
            qDs = [ D1 D2 ... Dn ] where A1 = V*D1*V' ,..., An =V*Dn*V'.
            qDs n x m x m array
    """
    As = As.copy()
    As = As.astype(float)
    n, m, _ = As.shape
    V = np.eye(m)
    
    encore = 1
    
    while encore:
        encore = 0
        for p in range(0, m-1):
            for q in range(p+1, m):
                part1 =  As[:, p, p] - As[:, q, q]
                part2 =  As[:, p, q] + As[:, q, p]
                g = np.stack([part1, part2])
                g = g @ g.T
                ton, toff = g[0, 0]-g[1, 1], g[0, 1]+g[1, 0]
                theta = 0.5*np.arctan2( toff , ton+np.sqrt(ton*ton+toff*toff) );
                c, s =np.cos(theta), np.sin(theta)
                
                if np.abs(s) > threshold:
                    encore = 1
                    cols_p = As[:, :, p].copy()
                    cols_q = As[:, :, q].copy()
                    As[:, :, p] = c * cols_p + s * cols_q
                    As[:, :, q] = c * cols_q - s * cols_p
                    
                    rows_p = As[:, p, :].copy()
                    rows_q = As[:, q, :].copy()
                    As[:, p, :] = c * rows_p + s * rows_q
                    As[:, q, :] = c * rows_q - s * rows_p
                    
                    temp = V[:, p].copy()
                    V[:, p] = c * temp + s * V[:, q]
                    V[:, q] = c * V[:, q] - s * temp
    Ds = As
    return V, Ds