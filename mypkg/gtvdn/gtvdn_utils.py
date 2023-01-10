# This files contains some utils for detection
# 1. Smooth with B-spline
# 2. Reduce the data dimension
# 3. estimate A matrix
import scipy
import numpy as np
from easydict import EasyDict as edict
from scipy.stats import multivariate_normal as mnorm
import rpy2.robjects as robj
from tqdm import trange
from utils.matrix import eig_sorted


# bw.nrd0 fn in R
def bwnrd0_py(x):
    hi = np.std(x, ddof=1)
    lo = np.min((hi, scipy.stats.iqr(x)/1.34))
    eps = 1e-10
    if np.abs(lo-0) <= eps:
        if np.abs(hi-0) > eps:
            lo = hi
        elif np.abs(x[0]-0) > eps:
            lo = x[0]
        else:
            lo = 1
    rev = 0.9 * lo * len(x)**(-0.2)
    return rev 

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
    This fn is to reduce the dim by Eig-decomposition
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
# for both eig and CPD
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


#
def lowrankA(Amat, rankKp=30):
    eigVals, eigVecs = eig_sorted(Amat)
    # if breaking conjugate eigval pair, add rankKp with 1
    if (eigVals[rankKp-1].imag + eigVals[rankKp].imag ) == 0:
        rankKp = rankKp + 1
    redAmat = np.matmul(np.matmul(eigVecs[:, :rankKp], np.diag(eigVals[:rankKp])), np.linalg.inv(eigVecs)[:rankKp, :])
    return redAmat.real



# Function to obtain the sum of Ai matrix for all MEG data
def get_Amats(dXmats, Xmats, timeSpan, downrate=1, fct=1, nRks=10, is_stack=False):
    """
    Input: 
        dXmats: The first derivative of Xmats, N x d x n matrix
        Xmats: Xmat, N x d x n matrix
        timeSpan: A list of time points with length n
        downrate: The downrate factor, determine how many Ai matrix to be summed
        nRks: the rank to keep when estimating the Amat for each data
        is_stack: stack the Amats for each subject or not
    Return:
        A d x d matrix, it is sum of N x n/downrate  Ai matrix
    """
    h = bwnrd0_py(timeSpan)*fct
    N, d, n = Xmats.shape
    Amats = []
    AmatsLow = []
    for ix in trange(N):
        Xmat, dXmat = Xmats[ix, :, :], dXmats[ix, :, :]
        if is_stack:
            curAmat = []
        else:
            curAmat = np.zeros((d, d))
        flag = 0
        for s in timeSpan[::downrate]:
            t_diff = timeSpan - s
            kernels = 1/np.sqrt(2*np.pi) * np.exp(-t_diff**2/2/h**2) # normal_pdf(x/h)
            kernelroot = kernels ** (1/2)
            kerdXmat = kernelroot[:, np.newaxis] * (dXmat.T) # n x d
            kerXmat = kernelroot[:, np.newaxis] * (Xmat.T) # n x d
            M = kerXmat.T.dot(kerXmat)/n
            XY = kerdXmat.T.dot(kerXmat)/n # it is Y\trans x X , formula is Amat = Y\trans X (X\trans X)^{-1}
            U, S, VT = np.linalg.svd(M)
            # Num of singular values to keep
            r = np.argmax(np.cumsum(S)/np.sum(S) >= 0.999) + 1 # For real data
            invM = VT.T[:, :r].dot(np.diag(1/S[:r])).dot(U.T[:r, :]) # M is symmetric and PSD
            if is_stack:
                curAmat.append( lowrankA( XY.dot(invM), nRks ) )
            else:
                curAmat = curAmat + XY.dot(invM)
            flag += 1
            
        #Amats.append(curAmat/flag)
        if is_stack:
            AmatsLow.append(curAmat)
        else:
            AmatsLow.append(lowrankA(curAmat/flag, nRks))
    return AmatsLow

