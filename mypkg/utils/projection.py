import numpy as np



def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex

    Solves the optimisation problem (using the algorithm from [1]):

        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 

    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project

    s: int, optional, default: 1,
       radius of the simplex

    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex

    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.

    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0]#[-1]
    if(len(rho) == 0):
        return 0
    rho = rho[-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w


def euclidean_proj_l1ball(v, s=1):
    """ Compute the Euclidean projection on a L1-ball

    Solves the optimisation problem (using the algorithm from [1]):

        min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s

    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project

    s: int, optional, default: 1,
       radius of the L1-ball

    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the L1-ball of radius s

    Notes
    -----
    Solves the problem by a reduction to the positive simplex case

    See also
    --------
    euclidean_proj_simplex
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # compute the vector of absolute values
    u = np.abs(v)
    # check if v is already a solution
    if u.sum() <= s:
        # L1-norm is <= s
        return v
    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    w = euclidean_proj_simplex(u, s=s)
    # compute the solution to the original problem on v
    w *= np.sign(v)
    return w


# --below three---
# I do not them exactly.
def euclidean_proj_l1ball_cplx(curVal, L):
    curValAbs = np.abs(curVal)
    redValAbs = euclidean_proj_l1ball(curValAbs.copy(), L)
    ratioValAbs = redValAbs/curValAbs
    redVal = curVal * ratioValAbs
    return redVal

def euclidean_proj_l1ball_cplxmat(curMat, L):
    curMatAbs = np.abs(curMat)
    rowNorms = np.sum(np.abs(curMatAbs), axis=1)
    redRowNorms = euclidean_proj_l1ball(rowNorms.copy(), L)
    rowNorms[rowNorms==0] = 1e-5 # avoid divided-by-0 error
    ratioVec = redRowNorms/rowNorms
    curMatRed = ratioVec.reshape(-1, 1) * curMat
    return curMatRed, redRowNorms

def euclidean_proj_l1ball_cplxmul(eigValsMat, L1):
    singleRedCurValList = []
    for ix in range(eigValsMat.shape[-1]):
        curVal = eigValsMat[:, ix]
        redCurVal = euclidean_proj_l1ball_Cplx(curVal.copy(), L1)
        singleRedCurValList.append(redCurVal)
    singleRedCurValMat = np.array(singleRedCurValList).T
    return singleRedCurValMat