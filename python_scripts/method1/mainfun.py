def GetAmatMul(dXmats, Xmats, timeSpan, downrate=1, fct=1):
    """
    Input: 
        dXmats: The first derivative of Xmats, N x d x n matrix
        Xmats: Xmat, N x d x n matrix
        timeSpan: A list of time points with length n
        downrate: The downrate factor, determine how many Ai matrix to be summed
    Return:
        A d x d matrix, it is sum of N x n/downrate  Ai matrix
    """
    h = pybwnrd0(timeSpan)*fct
    N, d, n = Xmats.shape
    Amats = []
    for ix in trange(N):
        Amat = np.zeros((d, d))
        Xmat, dXmat = Xmats[ix, :, :], dXmats[ix, :, :]
        curAmat = np.zeros((d, d))
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
            curAmat = curAmat + XY.dot(invM)
            Amat = Amat + curAmat
        Amats.append(Amat)
    return Amats

def GetNewDataMul(dXmats, Xmats, Amats, rRaw, is_full=False):
    """
    Input: 
        dXmats: The first derivative of Xmats, N x d x n matrix
        Xmats: Xmat, N x d x n matrix
        Amat: The A matrix to to eigendecomposition, d x d
        rRaw:    The rank of A matrix
              If r is decimal, the rank is the number of eigen values which account for 100r % of the total variance
              If r is integer, the r in algorithm can be r + 1 if r breaks the conjugate eigval pairs. 
        is_full: Where return full outputs or not
    Return: 
        nXmats, ndXmats, N x r x n 
    """
    N = len(Amats)
    fullRess = []
    for ix in trange(N):
        r = rRaw
        Amat = Amats[ix]
        Xmat, dXmat = Xmats[ix],  dXmats[ix]
        
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
        
        tXmat =  np.matmul(eigVecsInv[kpidxs, :], Xmat)
        tdXmat =  np.matmul(eigVecsInv[kpidxs, :], dXmat)
        nrow, n = tXmat.shape
        nXmat = np.zeros((r, n))
        ndXmat = np.zeros((r, n))
        # Now I change to real first, then imag
        # Note that for real eigval, we do not need imag part.
        nXmat[:nrow, :] = tXmat.real
        nXmat[nrow:, :] =  tXmat.imag[(np.abs(eigVals.imag)!=0)[kpidxs], :]
        ndXmat[:nrow, :] = tdXmat.real
        ndXmat[nrow:, :] =  tdXmat.imag[(np.abs(eigVals.imag)!=0)[kpidxs], :]
        fullRes = edict({"ndXmat":ndXmat, "nXmat":nXmat, "kpidxs":kpidxs, "eigVecs":eigVecs, "eigVals":eigVals, "r": r})
        fullRess.append(fullRes)
    return fullRess
def ScreeningMul(Ress, wh=10, showProgress=True):
    """
    Input:
        wh: screening window size
    """
    # Get the scanning stats at index k
    def GetScanStats(k, wh):
        lidx = k - wh + 1
        uidx = k + wh + 1

        pndXmatA = ndXmat[:, lidx:uidx]
        pnXmatA = nXmat[:, lidx:uidx]
        GamkA = GetGammak(pndXmatA, pnXmatA)
        nlogA = GetNlogk(pndXmatA, pnXmatA, GamkA)

        pndXmatL = ndXmat[:, lidx:(k+1)]
        pnXmatL = nXmat[:, lidx:(k+1)]
        GamkL = GetGammak(pndXmatL, pnXmatL)
        nlogL = GetNlogk(pndXmatL, pnXmatL, GamkL)

        pndXmatR = ndXmat[:, (k+1):uidx]
        pnXmatR = nXmat[:, (k+1):uidx]
        GamkR = GetGammak(pndXmatR, pnXmatR)
        nlogR = GetNlogk(pndXmatR, pnXmatR, GamkR)

        return nlogR + nlogL - nlogA

    N = len(Ress)
    _, n = Ress[0].ndXmat.shape
    canptss = []
    if showProgress:
        iterBar = trange(N, desc="Screening")
    else:
        iterBar = range(N)
    for ix in iterBar:
        ndXmat, nXmat = Ress[ix].ndXmat, Ress[ix].nXmat
        scanStats = []
        for iy in range(n):
            if iy < (wh-1):
                scanStats.append(np.inf)
            elif iy >= (n-wh):
                scanStats.append(np.inf)
            else:
                scanStats.append(GetScanStats(iy, wh))

        canpts = []
        for idx, scanStat in enumerate(scanStats):
            if (idx >= (wh-1)) and (idx < (n-wh)):
                lidx = idx - wh + 1
                uidx = idx + wh + 1
                if scanStat == np.min(scanStats[lidx:uidx]):
                    canpts.append(idx) # the change point is from 0 not 1

        canptss.append(canpts)
    return canptss
def get_gammak_4CPD(Ycur, Xcur):
    """
    Input: 
        Ycur: part of ndXmat, r x (j-i)
        Xcur: part of nXmat, r x (j-i)
    Return:
        Gamma matrix, r x r
    """
    r = Ycur.shape[0]
    GamMat = np.zeros((r, r))
    for ix in range(r):
        rY, rX = Ycur[ix, :], Xcur[ix, :]
        GamMat[ix, ix] = (rY.dot(rX))/(rX.dot(rX))
    return GamMat

def screening_4CPD(ndXmats, nXmats, wh=10, showProgress=True):
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
        GamkA = get_gammak_4CPD(pndXmatA, pnXmatA)
        nlogA = get_Nlogk(pndXmatA, pnXmatA, GamkA)

        pndXmatL = ndXmat[:, lidx:(k+1)]
        pnXmatL = nXmat[:, lidx:(k+1)]
        GamkL = get_gammak_4CPD(pndXmatL, pnXmatL)
        nlogL = get_Nlogk(pndXmatL, pnXmatL, GamkL)

        pndXmatR = ndXmat[:, (k+1):uidx]
        pnXmatR = nXmat[:, (k+1):uidx]
        GamkR = get_gammak_4CPD(pndXmatR, pnXmatR)
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
def dyna_prog_4CPD(ndXmat, nXmat, kappa, Lmin=None, canpts=None, maxM=None, is_full=False, Ms=None, showProgress=True):
    """
    Input:
    ndXmat: array, r x n. n is length of sequence. 
    nXmat: array, r x n. n is length of sequence. 
    kappa: The parameter of penalty
    Lmin: The minimal length between 2 change points
    canpts: candidate point set. list or array,  index should be from 1
    maxM: int, maximal number of change point 
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
        Gamk = get_gammak_4CPD(pndXmat, pnXmat)
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

    if (maxM is None) or (maxM>M0):
        maxM = M0 
    if not (Ms is None or len(Ms)==0):
        maxM = Ms[-1] if Ms[-1]>=maxM else maxM
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
    U = np.zeros(maxM+1) 
    U[0] = Hmat[0, -1]
    D = Hmat[:, -1]
    # contain the location of candidate points  (in python idx)
    Pos = np.zeros((M0+1, maxM)) + decon
    Pos[M0, :] = np.ones(maxM) * M0
    tau_mat = np.zeros((maxM, maxM)) + decon
    for k in range(maxM):
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
    U = U + 2*r*np.log(n)**kappa* (np.arange(1, maxM+2))
    chgMat = np.zeros(tau_mat.shape) + np.inf
    for iii in range(chgMat.shape[0]):
        idx = tau_mat[iii,: ]
        idx = np.array(idx[idx<np.inf], dtype=int)
        chgMat[iii, :(iii+1)]= np.array(canpts)[idx] + 1 
    
    mbic_numchg = np.argmin(U[:(maxM+1)])
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
def corr_plot():
    plt.figure(figsize=[15, 10])
    for ix, nam in enumerate(names):
        plt.subplot(2, 1, ix+1)
        curMat= corrMats[nam]
        sns.heatmap(curMat,  yticklabels=kysOrd, cmap="coolwarm", center=0, vmin=-1, vmax=1, annot=np.round(curMat, 2))
        plt.title(nam)
def heat_plot(idx, curFn):
    plt.figure(figsize=[20, 10])
    for ix, nam in enumerate(names):
        plt.subplot(1, 2, ix+1)
        curRes = CPDress[nam]
        sns.heatmap(curFn(curRes[idx]))
        plt.title(nam)
