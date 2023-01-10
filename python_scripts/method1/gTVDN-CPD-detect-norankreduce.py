#!/usr/bin/env python
# coding: utf-8

# In this file, we detect the switching pts without rank deduction. 
# 
# Here I do CPD such that both U and V are orthornormal

# In[1]:


import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

from pathlib import Path
from scipy.io import loadmat
from tqdm import trange, tqdm
from scipy.stats import ttest_ind
from collections import defaultdict as ddict
from easydict import EasyDict as edict
from mainfun import *
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


# import my own functions
import sys
sys.path.append("../../mypkg")
import importlib

# paras
import paras
importlib.reload(paras);
from paras import paras

# some useful constants
import constants
importlib.reload(constants)
from constants import REGION_NAMES, REGION_NAMES_WLOBE, RES_ROOT, DATA_ROOT


# In[6]:


# gtvdn
import gtvdn.gtvdn_post
importlib.reload(gtvdn.gtvdn_post)
from gtvdn.gtvdn_post import est_eigvals, update_kp

import gtvdn.gtvdn_pre
importlib.reload(gtvdn.gtvdn_pre)
from gtvdn.gtvdn_pre import preprocess_MEG

import gtvdn.gtvdn_utils
importlib.reload(gtvdn.gtvdn_utils)
from gtvdn.gtvdn_utils import get_bspline_est , get_newdata 


# In[8]:


# utils
import utils.matrix
importlib.reload(utils.matrix)
from utils.matrix import eig_sorted

import utils.misc
importlib.reload(utils.misc)
from utils.misc import paras2name, cumsum_cutoff, save_pkl, load_pkl

import utils.projection
importlib.reload(utils.projection)
from utils.projection import euclidean_proj_l1ball

import utils.standardize
importlib.reload(utils.standardize)
from utils.standardize import minmax, minmax_mat, minmax_pn

import utils.tensor
importlib.reload(utils.tensor)
from utils.tensor import decompose_three_way_orth, decompose_three_way_fix, sort_orthCPD

import utils.brain_plot
importlib.reload(utils.brain_plot)
from utils.brain_plot import reorder_U, U_2brain_vec


# ## Set parameters

# In[9]:


# in case you want to update any parameters
paras.keys()


# ## Load data

# In[12]:


datFil = list(DATA_ROOT.glob("70Ctrl*"))[0]
CtrlDat1 = loadmat(datFil)
CtrlDats = CtrlDat1["dk10"]


# In[13]:


datFil = list(DATA_ROOT.glob("87AD*"))[0]
ADDat1 = loadmat(datFil)
ADDats = ADDat1["dk10"]


# In[14]:


ADdatAdd = loadmat(DATA_ROOT/"DK_timecourse.mat")["DK_timecourse"]


# In[15]:


import mat73
CtrldatAdd = mat73.loadmat(DATA_ROOT/"timecourse_ucsfCONT_group.mat")["dk10"]


# In[16]:


baseDF = pd.read_csv(DATA_ROOT/"AllDataBaselineOrdered.csv")
CtrlKp = np.array(baseDF[baseDF["Grp"]=="Ctrl"]["KeepIt"] == 1)


# In[17]:


ADDatsAll = np.concatenate([ADDats, ADdatAdd[np.newaxis, :, :]], axis=0)
CtrlDatsAll = np.concatenate([CtrlDats, CtrldatAdd], axis=0)
CtrlDatsAll = CtrlDatsAll[CtrlKp]


# In[23]:


print(np.median(np.abs(ADDatsAll)), np.median(np.abs(CtrlDatsAll)))
print(np.mean(np.abs(ADDatsAll)), np.mean(np.abs(CtrlDatsAll)))


# In[22]:





# In[ ]:





# ## Validate the function (No need, correct)

# In[10]:


from tqdm import trange
from scipy.stats import multivariate_normal as mnorm


# Function to obtain the sum of Ai matrix




# In[20]:


YmatCtrl = prepMEG(CtrlDatsAll[:], paras)
YmatAD = prepMEG(ADDatsAll[:], paras)


# In[21]:


timeSpan = np.linspace(0, paras.T, YmatAD.shape[-1])

dXmatsAD, XmatsAD = GetBsplineEst(YmatAD, timeSpan, paras.lamb)
dXmatsCtrl, XmatsCtrl = GetBsplineEst(YmatCtrl, timeSpan, paras.lamb)


# In[ ]:


AmatCtrls = GetAmatMul(dXmatsCtrl, XmatsCtrl, timeSpan, downrate=paras.downRate, fct=paras.fct)
AmatADs = GetAmatMul(dXmatsAD, XmatsAD, timeSpan, downrate=paras.downRate, fct=paras.fct)


# In[68]:


ResCtrls = GetNewDataMul(dXmatsCtrl, XmatsCtrl, AmatCtrls, paras.r, True)
ResADs = GetNewDataMul(dXmatsAD, XmatsAD, AmatADs, paras.r, True)


# In[80]:


rksADs = [ix.ndXmat.shape[0] for ix in ResADs]
rksCtrls = [ix.ndXmat.shape[0] for ix in ResCtrls]
print(ttest_ind(rksADs, rksCtrls))
np.mean([rksADs, rksCtrls], axis=1)


# In[81]:


canptssADs = ScreeningMul(ResADs, wh=paras.wh, showProgress=True)
canptssCtrls = ScreeningMul(ResCtrls, wh=paras.wh, showProgress=True)


# In[82]:


detRessADs = []
for ix in trange(len(canptssADs)):
    res = EGenDy(ResADs[ix].ndXmat, ResADs[ix].nXmat, paras.kappa, Lmin=paras.Lmin, 
          canpts=canptssADs[ix], MaxM=paras.MaxM, is_full=True, 
          showProgress=False)
    detRessADs.append(res)


# In[83]:


detRessCtrls = []
for ix in trange(len(canptssCtrls)):
    res = EGenDy(ResCtrls[ix].ndXmat, ResCtrls[ix].nXmat, paras.kappa, Lmin=paras.Lmin, 
          canpts=canptssCtrls[ix], MaxM=paras.MaxM, is_full=True, 
          showProgress=False)
    detRessCtrls.append(res)


# In[136]:


curKp = 2.83
_, n = ResCtrls[0].ndXmat.shape
ecptssCtrls = np.array([UpdateKp(curKp, detRessCtrls[ix].U0, n, ResCtrls[ix].ndXmat.shape[0], paras) for ix in range(len(detRessCtrls))])
_, n = ResADs[0].ndXmat.shape
ecptssADs = np.array([UpdateKp(curKp, detRessADs[ix].U0, n, ResADs[ix].ndXmat.shape[0], paras) for ix in range(len(detRessADs))])
print(np.mean(ecptssADs), np.mean(ecptssCtrls))
print(ttest_ind(ecptssADs, ecptssCtrls, equal_var=False))

plt.hist(ecptssCtrls, alpha=0.5, label="Ctrl")
plt.hist(ecptssADs, alpha=0.5, label="AD")
plt.legend()


# In[ ]:





# ## Run 

# In[24]:


Ymat_ctrl = preprocess_MEG(CtrlDatsAll[:], paras)
Ymat_AD = preprocess_MEG(ADDatsAll[:], paras)


# In[ ]:


time_span = np.linspace(0, paras.T, Ymat_AD.shape[-1])

dXmats_AD, Xmats_AD = get_bspline_est(Ymat_AD, time_span, paras.lamb)
dXmats_ctrl, Xmats_ctrl = get_bspline_est(Ymat_ctrl, time_span, paras.lamb)

plt.figure(figsize=[10, 5])
plt.subplot(121)
for ix in range(68):
    plt.plot(Xmats_AD[0, ix, :])
plt.subplot(122)
for ix in range(68):
    plt.plot(Xmats_ctrl[0, ix, :])


# ### CP decomposition

# In[ ]:


Amat_ctrls_lowrank = get_Amats(dXmats_ctrl[:], Xmats_ctrl[:], time_span, downrate=paras.downsample_rate, 
                              fct=paras.fct, nRks=paras.num_ranks, is_sum=True)


# In[ ]:


Amat_ADs_lowrank = get_Amats(dXmats_AD[:], Xmats_AD[:], time_span, downrate=paras.downsample_rate, 
                            fct=paras.fct, nRks=paras.num_ranks, is_sum=True)


# In[ ]:


# rank-R decomposition
tensors_ctrl = np.transpose(np.array(Amat_ctrls_lowrank), (1, 2, 0))
tensors_AD = np.transpose(np.array(Amat_ADs_lowrank), (1, 2, 0))

# the initial value 
Amat_ctrl_lowrank = np.sum(Amat_ctrls_lowrank, axis=0)
U, _, VT = np.linalg.svd(Amat_ctrl_lowrank)
ctrl_CPD_init = [U[:, :paras.r], VT.T[:, :paras.r]]

CPDresult_ctrl = sort_orthCPD(decompose_three_way_orth(tensors_ctrl, paras.r, init=ctrl_CPD_init))
CPDresult_AD = decompose_three_way_fix(tensors_AD, init=[CPDresult_ctrl[0], CPDresult_ctrl[1]])


# In[41]:


names =["AD", "ctrl"]
CPDress = edict()
CPDress["AD"]= CPDresult_AD
CPDress["ctrl"]= CPDresult_ctrl


# In[ ]:





# #### CPD factors

# In[46]:


orgFn = lambda x: np.abs(x)
mmFn = lambda x: minmax_mat(np.abs(x), is_row)
cutFn = lambda x: minmax_mat(np.abs(x), is_row)>cutoff



# In[47]:


cutoff = 0.5
is_row = True
heat_plot(2, mmFn)
#heatPlotFn(2, mmFn)


# In[48]:


cutoff = 0.5
rksAD = cutFn(CPDress["AD"][-1]).sum(axis=1)
rksCtrl = cutFn(CPDress["ctrl"][-1]).sum(axis=1)
plt.figure(figsize=[10, 5])
plt.subplot(121)
_ = plt.boxplot([rksAD, rksCtrl], showfliers=False)
pval = ttest_ind(rksAD, rksCtrl, equal_var=False).pvalue
plt.xticks([1, 2], ["AD", "Ctrl"])
plt.title(f"Ranks: {np.mean(rksAD):.2f} vs {np.mean(rksCtrl):.2f} (Pvalue: {pval:.3f})")

plt.subplot(122)
plt.hist(rksAD, alpha=0.5, label="AD")
plt.hist(rksCtrl, alpha=0.5, label="Ctrl")
plt.legend()


# In[ ]:





# #### corrs with 7 networks

# In[49]:


import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns

defNetsFil = list(dataDir.glob("DK_dic68.csv"))[0]
defNets = pd.read_csv(defNetsFil).T

mmDefNets = {}
for ix in range(defNets.shape[-1]):
    curCol = defNets[ix]
    mmDefNets[curCol[0]] = minmax(np.array(curCol[1:]))
kysOrd = sorted(mmDefNets.keys())


# In[50]:


idx = 0
corrMats = edict()
for ix, nam in enumerate(names):
    curRes = CPDress[nam]
    curUs = np.abs(curRes[idx])
    corrMat = np.zeros((7, curUs.shape[-1]))
    for iy in range(curUs.shape[-1]):
        curU = minmax(curUs[:, iy])
        for iz, kz in enumerate(kysOrd):
            curV = mmDefNets[kz]
            corrMat[iz, iy] = pearsonr(curU, curV)[0]
    corrMats[nam] = corrMat


# In[51]:




# In[52]:


corr_plot()


# ### Reduce the dim of the data

# In[35]:


B1 = CPDresult_ctrl[0]
B2 = CPDresult_ctrl[1]


# In[63]:


ndXmats_ctrl =  np.matmul(B1.T[np.newaxis, :, :], dXmats_ctrl)
nXmats_ctrl =  np.matmul(B2.T[np.newaxis, :, :], Xmats_ctrl)
ndXmats_AD =  np.matmul(B1.T[np.newaxis, :, :], dXmats_AD)
nXmats_AD =  np.matmul(B2.T[np.newaxis, :, :], Xmats_AD)


# ### Screening

# In[68]:


# Function to calculate the  Gamma_k matrix during dynamic programming for CP Decomposition


# In[71]:




# In[76]:


candidate_pts_ctrls = screening_4CPD(ndXmats_ctrl, nXmats_ctrl, wh=paras.wh)
candidate_pts_ADs = screening_4CPD(ndXmats_AD, nXmats_AD, wh=paras.wh)


# ### Detection

# In[80]:


# Effcient dynamic programming to optimize the MBIC, 


# In[83]:


cpts_ctrls = []
for ix in trange(len(candidate_pts_ctrls)):
    res = dyna_prog_4CPD(ndXmats_ctrl[ix], nXmats_ctrl[ix], paras.kappa, Lmin=paras.Lmin,  canpts=candidate_pts_ctrls[ix], 
                     maxM=paras.maxM,  is_full=True,  showProgress=False)
    cpts_ctrls.append(res)


# In[84]:


cpts_ADs = []
for ix in trange(len(candidate_pts_ADs)):
    res = dyna_prog_4CPD(ndXmats_AD[ix], nXmats_AD[ix], paras.kappa, Lmin=paras.Lmin,  canpts=candidate_pts_ADs[ix], 
                     maxM=paras.maxM,  is_full=True,  showProgress=False)
    cpts_ADs.append(res)


# ### Save the results

# In[826]:


results = edict()
results.paras = paras

results.AD = edict()
results.AD.cpts = cpts_ADs
results.AD.candidate_pts = candidate_pts_ADs
results.AD.ndXmats = ndXmats_AD
results.AD.nXmats = nXmats_AD
results.AD.CPDres = CPDresult_AD

results.ctrl = edict()
results.ctrl.cpts = cpts_ctrls
results.ctrl.candidate_pts = candidate_pts_ctrls
results.ctrl.ndXmats = ndXmats_ctrl
results.ctrl.nXmats = nXmats_ctrl
results.ctrl.CPDres = CPDresult_ctrl
save_pkl(paras.res_dir/"CPD_results_detect_first.pkl", results)


# In[6]:


results = load_pkl(paras.res_dir/"CPD_results_detect_first.pkl")


# In[ ]:





# In[ ]:




