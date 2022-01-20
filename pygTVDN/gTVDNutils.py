import torch
import numpy as np
from easydict import EasyDict as edict
from tqdm.autonotebook import tqdm
import time
from Rfuns import bw_nrd0_R, smooth_spline_R



def euclidean_proj_simplexTorch(v, s=1):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and (torch.mean(v >= 0)==1):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = torch.sort(v,descending=True)[0]
    cssv = torch.cumsum(u, dim=0)
    # get the number of > 0 components of the optimal solution
    rho = torch.nonzero(u * torch.arange(1, n+1) > (cssv - s)).reshape(-1)#[-1]
    if(len(rho) == 0):
        return 0
    rho = rho[-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w


def euclidean_proj_l1ballTorch(v, s=1):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # compute the vector of absolute values
    u = torch.abs(v)
    # check if v is already a solution
    if u.sum() <= s:
        # L1-norm is <= s
        return v
    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    w = euclidean_proj_simplexTorch(u, s=s)
    # compute the solution to the original problem on v
    w *= torch.sign(v)
    return w


# ### Misc

# +
def genDiffMatfn(R, D):
    """
    Generate the Matrix for different of parameters, D_\mu and D_\nu, RD-R x RD
    """
    mat = torch.zeros(R*D-R, R*D, dtype=torch.int8)
    mat[:, R:] += torch.eye(R*D-R, dtype=torch.int8)
    mat[:, :-R] += -torch.eye(R*D-R, dtype=torch.int8)
    return mat.to_sparse()

def DiffMatOpt(Vec, R):
    """
    D_\mu operator, i.e. output is D_\mu Vec
    """ 
    VecDiff = Vec[R:] - Vec[:-R]
    return VecDiff

def DiffMatTOpt(Vec, R):
    """
    D_\mu\trans operator, i.e. output is D_\mu\trans Vec
    """ 
    outVec = torch.zeros(Vec.shape[0]+R)
    outVec[R:-R] = Vec[:-R] - Vec[R:]
    outVec[:R] = -Vec[:R]
    outVec[-R:] = Vec[-R:]
    return outVec
    

def genDiffMatSqfn(R, D):
    """
    Generate the squared Matrix for different of parameters, D_\mu \trans D_\nm, RD x RD
    """
    mVec = torch.ones(R*D, dtype=torch.int8)
    mVec[R:-R] += torch.ones(R*(D-2), dtype=torch.int8)
    mat = torch.diag(mVec)
    mat[R:, :-R] += -torch.eye(R*(D-1), dtype=torch.int8)
    mat[:-R, R:] += -torch.eye(R*(D-1), dtype=torch.int8)
    return mat.to_sparse()

def genIdenMatfn(s, R, D):
    """
    Generate the (R x RD) Matrix with only s^th (from 1) block is identity matrix 
    s: The index is from 1 not 0
    """
    assert s <= D, "Parameter s is to large!"
    s = s - 1
    mat = torch.zeros(R, R*D, dtype=torch.int8)
    mat[:, R*s:(R*s+R)] += torch.eye(R, dtype=torch.int8)
    return mat.to_sparse().double()

def colStackFn(a, R=None):
    """
    transform between R x D matrix and RD vector where the elements are stacked by columns
    
    args:
        a: The target to do transfrom
        R: num of rows
    """
    
    if a.dim() == 2:
        out = a.permute((1, 0)).reshape(-1)
    elif a.dim() == 1:
        out = a.reshape(-1, R).permute((1, 0))
    return out


def PreProcess(Ymats, decimateRate=10, is_detrend=True):
    """
    Input:
        Ymats: multiple datasets, N x d x n, N is the num of datasets
    """
    if decimateRate is not None:
        Ymats = decimate(Ymats, q=decimateRate, ftype="fir")
    if is_detrend:
        Ymats = detrend(Ymats)
    return Ymats


def GetBsplineEst(Ymat, time, lamb=1e-6, nKnots=None):
    """
    Input:
        Ymat: The observed data matrix, d x n
        time: A list of time points of length n
    return:
        The estimated Xmat and dXmat, both are d x n
    """
    d, n = Ymat.shape
    Xmatlist = []
    dXmatlist = []
    for i in range(d):
        spres = smooth_spline_R(x=time, y=Ymat[i, :], lamb=lamb, nKnots=nKnots)
        Xmatlist.append(spres["yhat"])
        dXmatlist.append(spres["ydevhat"])
    Xmat = np.array(Xmatlist)
    dXmat = np.array(dXmatlist)
    return dXmat, Xmat


def GetAmat(dXmat, Xmat, time, downrate=1, fct=1):
    """
    Input: 
        dXmat: The first derivative of Xmat, d x n matrix
        Xmat: Xmat, d x n matrix
        time: A list of time points with length n
        downrate: The downrate factor, determine how many Ai matrix to be summed
    Return:
        A d x d matrix, it is sum of n/downrate  Ai matrix
    """
    h = bw_nrd0_R(time, fct=fct)
    d, n = Xmat.shape
    Amat = np.zeros((d, d))
    for idx, s in enumerate(time[::downrate]):
        t_diff = time - s
        kernels = 1/np.sqrt(2*np.pi) * np.exp(-t_diff**2/2/h**2) # normal_pdf(x/h)
        kernelroot = kernels ** (1/2)
        kerdXmat = kernelroot[:, np.newaxis] * (dXmat.T) # n x d
        kerXmat = kernelroot[:, np.newaxis] * (Xmat.T) # n x d
        M = kerXmat.T.dot(kerXmat)/n
        XY = kerdXmat.T.dot(kerXmat)/n
        U, S, VT = np.linalg.svd(M)
        # Num of singular values to keep
        # r = np.argmax(np.cumsum(S)/np.sum(S) > 0.999) + 1 # For simulation
        r = np.argmax(np.cumsum(S)/np.sum(S) >= 0.999) + 1 # For real data
        invM = U[:, :r].dot(np.diag(1/S[:r])).dot(VT[:r, :])
        Amat = Amat + XY.dot(invM)
    return Amat


# Function to obtain the New Xmat and dXmat to do change point detection
def GetNewEst(dXmat, Xmat, Amat, r, is_full=False):
    """
    Input: 
        dXmat, Estimated first derivative of X(t), d x n
        Xmat, Estimated of X(t), d x n
        Amat: The A matrix to to eigendecomposition, d x d
        r: The number of eigen values to keep
        is_full: Where return full outputs or not
    Return: 
        nXmat, ndXmat, rAct x n 
    """
    _, n = dXmat.shape
    eigVals, eigVecs = np.linalg.eig(Amat)
    eigValsfull = np.concatenate([[np.Inf], eigVals])
    kpidxs = np.where(np.diff(np.abs(eigValsfull))[:r] != 0)[0]
    eigVecsInv = np.linalg.inv(eigVecs)
    nXmat = eigVecsInv[kpidxs, :].dot(Xmat)
    ndXmat = eigVecsInv[kpidxs, :].dot(dXmat)
    if is_full:
        return edict({"ndXmat":ndXmat, "nXmat":nXmat, "kpidxs":kpidxs, "eigVecs":eigVecs, "eigVals":eigVals, "r": r})
    else:
        return ndXmat, nXmat

# optimization class
class OneStepOpt():
    """
        I concatenate the real and image part into one vector.
    """
    def __init__(self, X, Y, lastTheta, penalty="SCAD", is_ConGrad=False, **paras):
        """
         Input: 
             Y: A matrix with shape, R x n, Complex
             X: A matrix with shape, R x n, 
             lastTheta: The parameters for optimizing at the last time step, initial parameters, vector of 2R(n-1), real data
             penalty: The penalty type, "SCAD" or "GroupLasso"
             is_ConGrad: Whether to use conjugate gradient method to update gamma or not. 
                        When data are large, it is recommended to use it
             paras:
                 beta: tuning parameter for iteration
                 alp: tuning parameter for iteration
                 rho: a vector of length (D-1)2R, real data
                 lam: the parameter for SCAD/Group lasso
                 a: the parameter for SCAD, > 1+1/beta
                 iterNum:  integer, number of iterations
                           if iterNum < 0, optimization without penalty.
                 iterC: decimal, stopping rule
                 eps: decimal, stopping rule for conjugate gradient method
                 b: radius of the L1-ball projection
        """
        
        parasDefVs = {"a": 2.7,  "beta": 1, "alp": 0.9,  "rho": None,  "lam": 1e2, 
                      "iterNum": 100, "iterC": 1e-4, "eps": 1e-6, "b": 100}
        
        self.paras = edict(parasDefVs)
        for key in paras.keys():
            self.paras[key] = paras[key]
        
            
            
        self.R, self.n = X.shape
        self.R2 = 2*self.R
        self.X = X
        self.Y = Y
        
        if self.paras.rho is None:
            self.paras.rho = torch.ones(self.R2*(self.n-1))
            
        self.lastTheta= lastTheta
        self.newVecGam = None
        self.halfRho = None
        self.penalty = penalty.lower()
        self.is_ConGrad = is_ConGrad
            
        self.leftMat = None
        self.leftMatVec = None
        self.NewXYR2 = None
        
        self.NewXr = X.real
        self.NewYr = Y.real
        self.NewXi = X.imag
        self.NewYi = Y.imag
        
        self.GamMat = None
        self.ThetaMat = None
        self.numEs = 1
        
        
    def _LeftMatOpt(self, vec):
        rVec1 = self.leftMatVecP1 * vec
        rVec2 = self.paras.beta * DiffMatTOpt(DiffMatOpt(vec, self.R2), self.R2)
        return rVec1 + rVec2
    
    def _ConjuGrad(self, vec, maxIter=1000):
        """ 
        Ax = vec
        """
        eps = self.paras.eps
        
        xk = torch.zeros_like(vec)
        rk = vec - self._LeftMatOpt(xk)
        pk = rk
        if torch.norm(rk) <= eps:
            return xk
        
        for k in range(maxIter):
            alpk = torch.sum(rk**2) / torch.sum(pk * self._LeftMatOpt(pk))
            xk = xk + alpk * pk 
            
            rk_1 = rk
            rk = rk - alpk * self._LeftMatOpt(pk)
            
            if torch.norm(rk) <= eps:
                break 
                
            betk = torch.sum(rk**2)/torch.sum(rk_1**2)
            pk = rk + betk * pk
            
        return xk
        
    
    def updateVecGam(self):
        """
            I use conjugate gradient to solve it. 
            Update the Gamma matrix, first step 
        """
        self.DiffMatSq = genDiffMatSqfn(self.R2, self.n) # R2n x R2n
        
        
        if self.leftMat is None:
            NewXSq = self.NewXr**2 + self.NewXi**2 # R x n
            NewXSqR2 = torch.cat((NewXSq, NewXSq), dim=0) # 2R x n
            self.leftMat = torch.diag(NewXSqR2.T.flatten()/self.numEs).to_sparse() +  \
                    self.paras.beta * self.DiffMatSq
        
        if self.NewXYR2 is None:
            NewXY1 = self.NewXr * self.NewYr + self.NewXi * self.NewYi # R x n
            NewXY2 = - self.NewXi * self.NewYr + self.NewXr * self.NewYi # R x n
            self.NewXYR2 = torch.cat((NewXY1, NewXY2), dim=0) # 2R x n
        rightVec = self.NewXYR2.T.flatten()/self.numEs + \
                    DiffMatTOpt(self.paras.rho + self.paras.beta * self.lastTheta, self.R2)
        
        self.newVecGam, _  = torch.solve(rightVec.reshape(-1, 1), self.leftMat.to_dense()) 
        self.newVecGam = self.newVecGam.reshape(-1)
        
    def updateVecGamConGra(self):
        """
            Update the Gamma matrix, first step, wth Conjugate Gradient Method
        """
        
        if self.leftMat is None:
            NewXSq = self.NewXr**2 + self.NewXi**2
            NewXSqR2 = torch.cat((NewXSq, NewXSq), dim=0) # 2R x n
            self.leftMatVecP1 = NewXSqR2.T.flatten()/self.numEs
        
        if self.NewXYR2 is None:
            NewXY1 = self.NewXr * self.NewYr + self.NewXi * self.NewYi
            NewXY2 = - self.NewXi * self.NewYr + self.NewXr * self.NewYi
            self.NewXYR2 = torch.cat((NewXY1, NewXY2), dim=0) # 2R x n
        rightVec = self.NewXYR2.T.flatten()/self.numEs + \
                    DiffMatTOpt(self.paras.rho + self.paras.beta * self.lastTheta, self.R2)
        
        self.newVecGam = self._ConjuGrad(rightVec)
    
    def projVecGam(self):
        """
        project Vec(Gam) in L1-ball with radius b based on mode as complex numbers
        """
        GamMat = colStackFn(self.newVecGam, self.R2)
        GamMatCplx = torch.complex(GamMat[:self.R, :], GamMat[self.R:, :])
        newVecGamMode = GamMatCplx.abs().T.flatten()
        projVecGamMode = euclidean_proj_l1ballTorch(newVecGamMode, self.paras.b)
        projGamMode = colStackFn(projVecGamMode, self.R)
        projGamMatCplx = GamMatCplx * projGamMode/GamMatCplx.abs()
        GamMat = torch.cat([projGamMatCplx.real, projGamMatCplx.imag])
        self.newVecGam = GamMat.T.flatten()
        
    def updateHRho(self):
        """
            Update the vector rho at 1/2 step, second step
        """
        halfRho = self.paras.rho - self.paras.alp * self.paras.beta * (DiffMatOpt(self.newVecGam, self.R2) - self.lastTheta)
        self.halfRho = halfRho
       
    
    def updateTheta(self):
        """
            Update the vector Theta, third step
        """
        halfTheta = DiffMatOpt(self.newVecGam, self.R2) - self.halfRho/self.paras.beta
        tranHTheta = halfTheta.reshape(-1, self.R2) # n-1 x 2R
        hThetaL2Norm = tranHTheta.abs().square().sum(axis=1).sqrt() # n - 1
        normCs = torch.zeros_like(hThetaL2Norm) - 1
        
        normC1 = hThetaL2Norm - self.paras.lam/self.paras.beta
        normC1[normC1<0] = 0
        
        normC2 = (self.paras.beta * (self.paras.a - 1) * hThetaL2Norm - self.paras.a * self.paras.lam)/(self.paras.beta * self.paras.a - self.paras.beta -1)
        
        c1 = (1+1/self.paras.beta)* self.paras.lam
        c2 = self.paras.a * self.paras.lam
        
        normCs[hThetaL2Norm<=c1] = normC1[hThetaL2Norm<=c1]
        normCs[hThetaL2Norm>c2] = hThetaL2Norm[hThetaL2Norm>c2]
        normCs[normCs==-1] = normC2[normCs==-1]
        
        normCs[normCs!=0] = normCs[normCs!=0]/hThetaL2Norm[normCs!=0]
        
        self.lastTheta = (tranHTheta*normCs.reshape(-1, 1)).flatten()
        
    def updateThetaGL(self):
        """
            Update the vector Theta, third step with group lasso penalty
        """
        halfTheta = DiffMatOpt(self.newVecGam, self.R2) - self.halfRho/self.paras.beta
        tranHTheta = halfTheta.reshape(-1, self.R2) # n-1 x 2R
        hThetaL2Norm = tranHTheta.abs().square().sum(axis=1).sqrt() # n-1
        
        normC1 = hThetaL2Norm - self.paras.lam
        normC1[normC1<0] = 0
        
        normCs = normC1
        
        normCs[normC1!=0] = normC1[normC1!=0]/hThetaL2Norm[normC1!=0]
        self.lastTheta = (tranHTheta*normCs.reshape(-1, 1)).flatten()
        
    
    def updateRho(self):
        """
            Update the vector rho, fourth step
        """
        newRho = self.halfRho - self.paras.alp * self.paras.beta * (DiffMatOpt(self.newVecGam, self.R2) - self.lastTheta)
        self.paras.rho = newRho
        
        
    def OneLoop(self):
        """
        Run one loop for the opt
        """
        
        ts = []
        ts.append(time.time())
        if self.is_ConGrad:
            self.updateVecGamConGra()
        else:
            self.updateVecGam()
            
        ts.append(time.time())
            
        self.projVecGam()
        
        ts.append(time.time())
        
        self.updateHRho()
        ts.append(time.time())
        
        if self.penalty.startswith("scad"):
            self.updateTheta()
        elif self.penalty.startswith("group"):
            self.updateThetaGL()
        ts.append(time.time())
        
        self.updateRho()
        ts.append(time.time())
        #print(np.diff(ts))
        
    
    def __call__(self, is_showProg=False, leave=False, **paras):
        for key in paras.keys():
            self.paras[key] = paras[key]
        if self.paras.iterC is None:
            self.paras.iterC = 0
        
        chDiff = torch.tensor(1e10)
        self.OneLoop()
        
        lastVecGam = self.newVecGam
        if is_showProg:
            with tqdm(total=self.paras.iterNum, leave=leave) as pbar:
                for i in range(self.paras.iterNum):
                    pbar.set_description(f"Inner Loop: The chdiff is {chDiff.item():.3e}.")
                    pbar.update(1)
                    
                    self.OneLoop()
                    
                    chDiff = torch.norm(self.newVecGam-lastVecGam)/torch.norm(lastVecGam)
                    lastVecGam = self.newVecGam
                    if chDiff < self.paras.iterC:
                        pbar.update(self.paras.iterNum)
                        break
        else:
            for i in range(self.paras.iterNum):
                
                self.OneLoop()
                
                chDiff = torch.norm(self.newVecGam-lastVecGam)/torch.norm(lastVecGam)
                lastVecGam = self.newVecGam
                if chDiff < self.paras.iterC:
                    break
        
        self._post()
            
    def _post(self):
        self.GamMat = colStackFn(self.newVecGam, self.R2) # 2R x n
        self.ThetaMat = colStackFn(self.lastTheta, self.R2) # 2R x (n-1)