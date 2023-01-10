from easydict import EasyDict as edict
import numpy as np
from pathlib import Path

paras = edict()
paras.is_detrend = True
paras.decimate_rate = 5
paras.T = 2 # the total time course
paras.lamb = 1e-4 # the smooth parameter for smooth spline
paras.fct = 0.5 #fct: The factor to adjust h when estimating A matrix
paras.downsample_rate = 20#the downsample factor, determine how many Ai matrix to contribute to estimate the eigen values/vectors
paras.r = 20 #    r: The rank of A matrix, 
             # If r is decimal, the rank is the number of eigen values which account for 100r % of the total variance
             # If r is integer, the r in algorithm can be r + 1 if r breaks the conjugate eigval pairs. 
paras.Lmin = 200 # Lmin: The minimal length between 2 change points
paras.maxM = 20 #  MaxM: int, maximal number of change point 
paras.kappa = 3.210 #kappa: The parameter of penalty in MBIC
paras.wh = 20 # screening window size
paras.kps =  np.linspace(1, 4, 1000)
paras.L = 1e2 # the  L1-ball projectoin radius
paras.num_ranks = 10 # the rank to keep when estimating the Amat for each data
paras.cutoff = 0.6
paras.Cs = np.array([100, 25, 10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.01])
paras.penalty = "l2"

paras.data_dir = Path("../data")
paras.res_dir = Path("../results")
paras.fig_dir = Path("../figs")

paras.cur_dir = Path(__file__).parent