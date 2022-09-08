import numpy as np
from tqdm import trange
from scipy.stats import multivariate_normal as mnorm
import rpy2.robjects as robj
from tqdm import trange
from easydict import EasyDict as edict


#  smooth spline in R
def smooth_spline_R(x, y, lamb, nKnots=None):
    smooth_spline_f = robj.r["smooth.spline"]
    x_r = robj.FloatVector(x)
    y_r = robj.FloatVector(y)
    if nKnots is None:
        args = {"x": x_r, "y": y_r, "lambda": lamb}
    else:
        args = {"x": x_r, "y": y_r, "lambda": lamb, "nknots":nKnots}
    spline = smooth_spline_f(**args)
    ysp = np.array(robj.r['predict'](spline, deriv=0).rx2('y'))
    ysp_dev1 = np.array(robj.r['predict'](spline, deriv=1).rx2('y'))
    return {"yhat": ysp, "ydevhat": ysp_dev1}


# Function to obtain the Bspline estimate of Xmats and dXmats, N x d x n
def get_bspline_est(Ymats, timeSpan, lamb=1e-6, nKnots=None):
    """
    Input:
        Ymats: The observed data matrix, N x d x n
        timeSpan: A list of time points of length n
        lamb: the smooth parameter, the larger the smoother. 
    return:
        The estimated Xmats and dXmats, both are N x d x n
    """
    N, d, n = Ymats.shape
    Xmats = np.zeros((N, d, n))
    dXmats = np.zeros((N, d, n))
    for ix in trange(N):
        for iy in range(d):
            spres = smooth_spline_R(x=timeSpan, y=Ymats[ix, iy, :], lamb=lamb, nKnots=nKnots)
            Xmats[ix, iy, :] = spres["yhat"]
            dXmats[ix, iy, :] = spres["ydevhat"]
    return dXmats, Xmats


def get_newdata(dXmats, Xmats, Amat, r, is_full=False):
    """
    Input: 
        dXmats: The first derivative of Xmats, N x d x n matrix
        Xmats: Xmat, N x d x n matrix
        Amat: The A matrix to to eigendecomposition, d x d
        r:    The rank of A matrix
              If r is decimal, the rank is the number of eigen values which account for 100r % of the total variance
              If r is integer, the r in algorithm can be r + 1 if r breaks the conjugate eigval pairs. 
        is_full: Where return full outputs or not
    Return: 
        nXmats, ndXmats, N x r x n 
    """
    eigVals, eigVecs = np.linalg.eig(Amat)
    # sort the eigvs and eigvecs
    sidx = np.argsort(-np.abs(eigVals))
    eigVals = eigVals[sidx]
    eigVecs = eigVecs[:, sidx]
    if r is None:
        rSel = np.where(np.cumsum(np.abs(eigVals))/np.sum(np.abs(eigVals)) >0.8)[0][0] + 1
        r = rSel
    elif r < 1:
        rSel = np.where(np.cumsum(np.abs(eigVals))/np.sum(np.abs(eigVals)) >r)[0][0] + 1
        r = rSel
        
    # if breaking conjugate eigval pair, add r with 1
    if (eigVals[r-1].imag + eigVals[r].imag ) == 0:
        r = r + 1

    eigValsfull = np.concatenate([[np.Inf], eigVals])
    kpidxs = np.where(np.diff(np.abs(eigValsfull))[:r] != 0)[0]
    eigVecsInv = np.linalg.inv(eigVecs)
    
    tXmats =  np.matmul(eigVecsInv[np.newaxis, kpidxs, :], Xmats)
    tdXmats =  np.matmul(eigVecsInv[np.newaxis, kpidxs, :], dXmats)
    N, nrow, n = tXmats.shape
    nXmats = np.zeros((N, r, n))
    ndXmats = np.zeros((N, r, n))
    # Now I change to real first, then imag
    # Note that for real eigval, we do not need imag part.
    nXmats[:, :nrow, :] = tXmats.real
    nXmats[:, nrow:, :] =  tXmats.imag[:,(np.abs(eigVals.imag)!=0)[kpidxs], :]
    ndXmats[:, :nrow, :] = tdXmats.real
    ndXmats[:, nrow:, :] =  tdXmats.imag[:,(np.abs(eigVals.imag)!=0)[kpidxs], :]
    if is_full:
        return edict({"ndXmats":ndXmats, "nXmats":nXmats, "kpidxs":kpidxs, "eigVecs":eigVecs, "eigVals":eigVals, "r": r})
    else:
        return ndXmats, nXmats
    


# Function to calculate the negative log likelihood during dynamic programming
def get_Nlogk(pndXmat, pnXmat, Gamk):
    """
    Input: 
        pndXmat: part of ndXmat, rAct x (j-i)
        pnXmat: part of nXmat, rAct x (j-i)
        Gamk: Gamma matrix, rAct x rAct
    Return:
        The Negative log likelihood
    """
    _, nj = pndXmat.shape
    resd = pndXmat - Gamk.dot(pnXmat)
    SigMat = resd.dot(resd.T)/nj
    U, S, VT = np.linalg.svd(SigMat)
    kpidx = np.where(S > (S[0]*1.490116e-8))[0]
    newResd = (U[:, kpidx].T.dot(resd)).T
    meanV = np.zeros(newResd.shape[1])
    Nloglike = - mnorm.logpdf(newResd, mean=meanV, cov=np.diag(S[kpidx])).sum()
    return Nloglike


# Function to calculate the  Gamma_k matrix during dynamic programming
def get_gammak(Ycur, Xcur, kpidxs):
    """
    Input: 
        pndXmat: part of ndXmat, r x (j-i)
        pnXmat: part of nXmat, r x (j-i)
        kpidxs: the kpidxs in getnewdata fn
    Return:
        Gamma matrix, r x r
    """
    r = Ycur.shape[0]
    GamMat = np.zeros((r, r))
    for iy in range(len(kpidxs)):
        if iy < (len(kpidxs)-1):
            is_real = (kpidxs[iy+1]-kpidxs[iy])==1
        else:
            is_real = kpidxs[iy] == (r-1)
        rY, rX = Ycur[iy, :], Xcur[iy, :]
        if is_real:
            GamMat[iy, iy] = (rY.dot(rX))/(rX.dot(rX))
        else:
            # two vec
            idxCpl = iy +1 - np.sum(np.diff(kpidxs)[:iy] == 1) # the ordinal number of complex number
            iidx = len(kpidxs) + idxCpl - 1
            iY, iX = Ycur[iidx, :], Xcur[iidx, :]
            den = iX.dot(iX) + rX.dot(rX)
            a = (rX.dot(rY) + iX.dot(iY))/den
            b = (rX.dot(iY) - iX.dot(rY))/den
            GamMat[iy, iy] = a
            GamMat[iidx, iidx] = a
            GamMat[iy, iidx] = -b
            GamMat[iidx, iy] = b
    return GamMat
    

    # Obtain the candidate point set via screening
def screening(ndXmats, nXmats, kpidxs, wh=10, showProgress=True):
    """
    Input:
        wh: screening window size
    """
    # Get the scanning stats at index k
    def _get_scan_stats(k, wh):
        lidx = k - wh + 1
        uidx = k + wh + 1

        pndXmatA = ndXmat[:, lidx:uidx]
        pnXmatA = nXmat[:, lidx:uidx]
        GamkA = get_gammak(pndXmatA, pnXmatA, kpidxs)
        nlogA = get_Nlogk(pndXmatA, pnXmatA, GamkA)

        pndXmatL = ndXmat[:, lidx:(k+1)]
        pnXmatL = nXmat[:, lidx:(k+1)]
        GamkL = get_gammak(pndXmatL, pnXmatL, kpidxs)
        nlogL = get_Nlogk(pndXmatL, pnXmatL, GamkL)

        pndXmatR = ndXmat[:, (k+1):uidx]
        pnXmatR = nXmat[:, (k+1):uidx]
        GamkR = get_gammak(pndXmatR, pnXmatR, kpidxs)
        nlogR = get_Nlogk(pndXmatR, pnXmatR, GamkR)

        return nlogR + nlogL - nlogA

    N, rAct, n = ndXmats.shape
    canptss = []
    if showProgress:
        iterBar = trange(N, desc="Screening")
    else:
        iterBar = range(N)
    for ix in iterBar:
        ndXmat, nXmat = ndXmats[ix, :, :], nXmats[ix, :, :]
        scanStats = []
        for iy in range(n):
            if iy < (wh-1):
                scanStats.append(np.inf)
            elif iy >= (n-wh):
                scanStats.append(np.inf)
            else:
                scanStats.append(_get_scan_stats(iy, wh))

        canpts = []
        for idx, scanStat in enumerate(scanStats):
            if (idx >= (wh-1)) and (idx < (n-wh)):
                lidx = idx - wh + 1
                uidx = idx + wh + 1
                if scanStat == np.min(scanStats[lidx:uidx]):
                    canpts.append(idx) # the change point is from 0 not 1

        canptss.append(canpts)
    return canptss


# Effcient dynamic programming to optimize the MBIC, 
def dyna_prog(ndXmat, nXmat, kappa, kpidxs, Lmin=None, canpts=None, MaxM=None, is_full=False, Ms=None, showProgress=True):
    """
    Input:
    ndXmat: array, r x n. n is length of sequence. 
    nXmat: array, r x n. n is length of sequence. 
    kappa: The parameter of penalty
    Lmin: The minimal length between 2 change points
    canpts: candidate point set. list or array,  index should be from 1
    MaxM: int, maximal number of change point 
    Ms: the list containing prespecified number of change points.
       When Ms=None, it means using MBIC to determine the number of change points
    is_full: Where return full outputs or not
    Return:
        change point set with index starting from 1
        chgMat: A matrix containing the change points for each number of change point
        U0: MBIC without penalty
        U:  MBIC  for each number of change point
    """
    def _nloglk(i, j):
        length = j - i + 1
        pndXmat = ndXmat[:, i:(j+1)]
        pnXmat = nXmat[:, i:(j+1)]
        Gamk = get_gammak(pndXmat, pnXmat, kpidxs)
        if length >= Lmin:
            return get_Nlogk(pndXmat, pnXmat, Gamk)
        else:
            return decon 

    r, n = nXmat.shape
    if Lmin is None:
        Lmin = r
        
    decon = np.inf

    if Ms is not None:
        Ms = sorted(Ms)
    if canpts is None:
        canpts = np.arange(n-1)
    else:
        canpts = np.array(canpts)
    M0 = len(canpts) # number of change point in candidate point set

    if (MaxM is None) or (MaxM>M0):
        MaxM = M0 
    if not (Ms is None or len(Ms)==0):
        MaxM = Ms[-1] if Ms[-1]>=MaxM else MaxM
    canpts_full = np.concatenate(([-1], canpts, [n-1]))
    canpts_full2 = canpts_full[1:]
    canpts_full1 = canpts_full[:-1] + 1 # small

    Hmat = np.zeros((M0+1, M0+1)) + decon

    # create a matrix 
    if showProgress:
        proBar = trange(M0+1, desc="Dynamic Programming")
    else:
        proBar = range(M0+1)
    for ix in proBar:
        for jx in range(ix, M0+1):
            iidx, jjdx = canpts_full1[ix],  canpts_full2[jx]
            Hmat[ix, jx]  = _nloglk(iidx, jjdx)

    # vector contains results for each number of change point
    U = np.zeros(MaxM+1) 
    U[0] = Hmat[0, -1]
    D = Hmat[:, -1]
    # contain the location of candidate points  (in python idx)
    Pos = np.zeros((M0+1, MaxM)) + decon
    Pos[M0, :] = np.ones(MaxM) * M0
    tau_mat = np.zeros((MaxM, MaxM)) + decon
    for k in range(MaxM):
        for j in range(M0): # n = M0 + 1
            dist = Hmat[j, j:-1] + D[(j+1):]
            #print(dist)
            D[j] = np.min(dist)
            Pos[j, 0] = np.argmin(dist) + j + 1
            if k > 0:
                Pos[j, 1:(k+1)] = Pos[int(Pos[j, 0]), 0:k]
        U[k+1] = D[0]
        tau_mat[k, 0:(k+1)] = Pos[0, 0:(k+1)] - 1
    U0 = U 
    U = U + 2*r*np.log(n)**kappa* (np.arange(1, MaxM+2))
    chgMat = np.zeros(tau_mat.shape) + np.inf
    for iii in range(chgMat.shape[0]):
        idx = tau_mat[iii,: ]
        idx = np.array(idx[idx<np.inf], dtype=int)
        chgMat[iii, :(iii+1)]= np.array(canpts)[idx] + 1 
    
    mbic_numchg = np.argmin(U[:(MaxM+1)])
    if mbic_numchg == 0:
        mbic_ecpts = np.array([])
    else:
        idx = tau_mat[int(mbic_numchg-1),: ]
        idx = np.array(idx[idx<np.inf], dtype=int)
        mbic_ecpts = np.array(canpts)[idx] + 1
        
    if Ms is None or len(Ms)==0:
        if not is_full:
            return edict({"U":U, "mbic_ecpts": mbic_ecpts})
        else:
            return edict({"U":U, "mbic_ecpts": mbic_ecpts, "chgMat": chgMat, "U0":U0})
    else:
        ecptss = []
        for numchg in Ms:
            if numchg == 0:
                ecpts = np.array([])
            else:
                idx = tau_mat[int(numchg-1),: ]
                idx = np.array(idx[idx<np.inf], dtype=int)
                ecpts = np.array(canpts)[idx] + 1
            ecptss.append(ecpts)
        if not is_full:
            return edict({"U":U, "ecptss": ecptss, "mbic_ecpts": mbic_ecpts})
        else:
            return edict({"U":U, "ecptss": ecptss, "mbic_ecpts": mbic_ecpts, "chgMat": chgMat, "U0":U0})