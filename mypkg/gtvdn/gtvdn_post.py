import numpy as np

# return the num of cpts under specific kappa
def update_kp(kappa, U0, n, r, paras):
    """
        n: the length of seq
        r: the dim of nXmats
    """
    U = U0 + 2*r*np.log(n)**kappa* (np.arange(1, paras.maxM+2))
    return np.argmin(U)

# Get the sigvals for each segment
def est_singular_vals(ecpts, ndXmat, nXmat):
    """
    Input: 
        ecpts: Estimated change points, 
        ndXmat: a rAct x n matrix
        nXmat: a rAct x n matrix
    Return:
        Estimated Singularvals, r x (len(ecpts) + 1) 
    """
    r, n = ndXmat.shape
    ecptsfull = np.concatenate(([0], ecpts, [n])) - 1
    ecptsfull = ecptsfull.astype(int)
    numchgfull = len(ecptsfull)
    
    def _obtain_singular_vals(Ycur, Xcur):
        lams = np.zeros(r) + np.inf
        for iy in range(r):
            rY, rX = Ycur[iy, :], Xcur[iy, :]
            lam = (rY.dot(rX))/(rX.dot(rX))
            lams[iy] = lam
        return lams

    ResegS = np.zeros((numchgfull-1, r))
    for  itr in range(numchgfull-1):
        lower = ecptsfull[itr] + 1
        upper = ecptsfull[itr+1] + 1
        Ycur = ndXmat[:, lower:upper]
        Xcur = nXmat[:, lower:upper]
        ResegS[itr, :] = _obtain_singular_vals(Ycur, Xcur)
    
    return ResegS.T

# Get the eigvals for each segment
def est_eigvals(ecpts, ndXmat, nXmat, kpidxs):
    """
    Input: 
        ecpts: Estimated change points, 
        ndXmat: a rAct x n matrix
        nXmat: a rAct x n matrix
        kpidxs: The intermedian output when calculating ndXmat, nXmat
    Return:
        Estimated Eigvals, r x (len(ecpts) + 1) 
    """
    r, n = ndXmat.shape
    ecptsfull = np.concatenate(([0], ecpts, [n])) - 1
    ecptsfull = ecptsfull.astype(int)
    numchgfull = len(ecptsfull)
    
    def _obtain_eigvals(Ycur, Xcur):
        lams = np.zeros(r, dtype=complex) + np.inf
        for iy in range(len(kpidxs)):
            if iy < (len(kpidxs)-1):
                is_real = (kpidxs[iy+1]-kpidxs[iy])==1
            else:
                is_real = kpidxs[iy] == (r-1)
            rY, rX = Ycur[iy, :], Xcur[iy, :]
            if is_real:
                # one vec
                a = (rY.dot(rX))/(rX.dot(rX))
                b = 0
            else:
                # two vec
                idxCpl = iy +1 - np.sum(np.diff(kpidxs)[:iy] == 1) # the ordinal number of complex number
                iidx = len(kpidxs) + idxCpl - 1
                iY, iX = Ycur[iidx, :], Xcur[iidx, :]
                den = iX.dot(iX) + rX.dot(rX)
                a = (rX.dot(rY) + iX.dot(iY))/den
                b = (rX.dot(iY) - iX.dot(rY))/den
            lams[kpidxs[iy]] = a + b*1j
        tmpIdx = np.where(lams==np.inf)[0]
        lams[tmpIdx] = np.conjugate(lams[tmpIdx-1])
        return lams

    ResegS = np.zeros((numchgfull-1, r), dtype=complex)
    for  itr in range(numchgfull-1):
        lower = ecptsfull[itr] + 1
        upper = ecptsfull[itr+1] + 1
        Ycur = ndXmat[:, lower:upper]
        Xcur = nXmat[:, lower:upper]
        ResegS[itr, :] = _obtain_eigvals(Ycur, Xcur)
    
    return ResegS.T