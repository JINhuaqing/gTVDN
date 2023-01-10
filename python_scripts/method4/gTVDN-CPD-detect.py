#!/usr/bin/env python
# coding: utf-8

# In this file, we detect the switching pts without rank deduction. 
# 
# For CPD, we sum the Amat for each subject and
# 
# do orthogonalization for both B1 (U) and B2 (V).
# 
# But now, I estimate B1, B2 for both cohorts

# In[1]:


import scipy
import mat73
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from scipy.io import loadmat
from tqdm import trange, tqdm
from functools import partial
from pprint import pprint

from collections import defaultdict as ddict
from easydict import EasyDict as edict


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


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
from constants import REGION_NAMES, REGION_NAMES_WLOBE, RES_ROOT, DATA_ROOT, FIG_ROOT


# In[5]:


# gtvdn
import gtvdn.gtvdn_det_CPD
importlib.reload(gtvdn.gtvdn_det_CPD)
from gtvdn.gtvdn_det_CPD import screening_4CPD, dyna_prog_4CPD

import gtvdn.gtvdn_post
importlib.reload(gtvdn.gtvdn_post)
from gtvdn.gtvdn_post import update_kp

import gtvdn.gtvdn_pre
importlib.reload(gtvdn.gtvdn_pre)
from gtvdn.gtvdn_pre import preprocess_MEG

import gtvdn.gtvdn_utils
importlib.reload(gtvdn.gtvdn_utils)
from gtvdn.gtvdn_utils import get_bspline_est , get_newdata, get_Amats, get_Nlogk


# In[6]:


# utils
import utils.matrix
importlib.reload(utils.matrix)
from utils.matrix import eig_sorted

import utils.misc
importlib.reload(utils.misc)
from utils.misc import paras2name, cumsum_cutoff, save_pkl_dict2folder, load_pkl_folder2dict

import utils.projection
importlib.reload(utils.projection)
from utils.projection import euclidean_proj_l1ball

import utils.standardize
importlib.reload(utils.standardize)
from utils.standardize import minmax, minmax_mat, minmax_pn

import utils.tensor
importlib.reload(utils.tensor)
from utils.tensor import decompose_three_way_orth, sort_orthCPD

import utils.brain_plot
importlib.reload(utils.brain_plot)
from utils.brain_plot import reorder_U, U_2brain_vec


# In[ ]:





# ## Some fns

# In[20]:


# truncate small value in vec
def _cumsum_trunc(vec, cutoff=0.9):
    vec = vec.copy()
    idxs = cumsum_cutoff(vec, cutoff)
    counter_idxs = np.delete(np.arange(len(vec)), idxs)
    vec[counter_idxs] = 0
    return vec


# In[21]:


# plot the corrmat with 7 canonical nets
def _corr_plot(vecs, cutoff=0.05, trun_fn=lambda x:x, trans_fn=np.abs):
    vecs = np.array(vecs)
    if vecs.shape[0] != 68:
        vecs = vecs.T
    assert vecs.shape[0] == 68
    corrMat = np.zeros((7, vecs.shape[-1]))
    for ix in range(vecs.shape[-1]):
        curU = vecs[:, ix]
        curU = trun_fn(curU)
        for iy, kz in enumerate(_paras.canon_net_names):
            curV = _paras.canon_nets[kz]
            curV = trun_fn(curV)
            corr_v, pv = scipy.stats.pearsonr(curU, curV)
            if pv <= cutoff:
                corrMat[iy, ix] = corr_v
            else:
                corrMat[iy, ix] = 0
            
    plt.figure(figsize=[15, 5])
    trans_corrMat = trans_fn(corrMat)
    sns.heatmap(trans_corrMat,  yticklabels=_paras.canon_net_names, 
                cmap="coolwarm", center=0, 
                vmin=-1, vmax=1, annot=np.round(trans_corrMat, 2))
    return corrMat


# ## Set parameters

# In[9]:


pprint(paras)


# In[10]:


# in case you want to update any parameters
paras.keys()


# In[11]:


# this parameters only for this file
_paras = edict()
_paras.folder_name = "method4"
_paras.save_dir = RES_ROOT/_paras.folder_name
print(f"Save to {_paras.save_dir}")


# In[51]:


# load results
cur_res = load_pkl_folder2dict(_paras.save_dir)


# In[ ]:





# ## Load data

# In[13]:


datFil = list(DATA_ROOT.glob("70Ctrl*"))[0]
CtrlDat1 = loadmat(datFil)
CtrlDats = CtrlDat1["dk10"]


# In[14]:


datFil = list(DATA_ROOT.glob("87AD*"))[0]
ADDat1 = loadmat(datFil)
ADDats = ADDat1["dk10"]


# In[15]:


ADdatAdd = loadmat(DATA_ROOT/"DK_timecourse.mat")["DK_timecourse"]


# In[16]:


CtrldatAdd = mat73.loadmat(DATA_ROOT/"timecourse_ucsfCONT_group.mat")["dk10"]


# In[17]:


baseDF = pd.read_csv(DATA_ROOT/"AllDataBaselineOrdered.csv")
CtrlKp = np.array(baseDF[baseDF["Grp"]=="Ctrl"]["KeepIt"] == 1)


# In[18]:


ADDatsAll = np.concatenate([ADDats, ADdatAdd[np.newaxis, :, :]], axis=0)
CtrlDatsAll = np.concatenate([CtrlDats, CtrldatAdd], axis=0)
CtrlDatsAll = CtrlDatsAll[CtrlKp]


# In[19]:


defNetsFil = list(DATA_ROOT.glob("DK_dic68.csv"))[0]
defNets = pd.read_csv(defNetsFil).T

DefNets_dict = {}
for ix in range(defNets.shape[-1]):
    curCol = defNets[ix]
    DefNets_dict[curCol[0]] = np.array(curCol[1:])
net_names = sorted(DefNets_dict.keys())

_paras.canon_nets = DefNets_dict
_paras.canon_net_names = net_names


# In[ ]:





# ## Run 

# ### Bspline smooth

# In[18]:


Ymat_ctrl = preprocess_MEG(CtrlDatsAll[:], paras)
Ymat_AD = preprocess_MEG(ADDatsAll[:], paras)


# In[19]:


time_span = np.linspace(0, paras.T, Ymat_AD.shape[-1])

if not ("dXXmats_AD" in cur_res.keys()):
    dXmats_AD, Xmats_AD = get_bspline_est(Ymat_AD, time_span, paras.lamb)
    dXmats_ctrl, Xmats_ctrl = get_bspline_est(Ymat_ctrl, time_span, paras.lamb)
    cur_res.dXXmats_AD = [dXmats_AD, Xmats_AD]
    cur_res.dXXmats_ctrl = [dXmats_ctrl, Xmats_ctrl]
    save_pkl_dict2folder(RES_ROOT/_paras.folder_name, cur_res)

plt.figure(figsize=[10, 5])
plt.subplot(121)
for ix in range(68):
    plt.plot(Xmats_AD[0, ix, :])
plt.subplot(122)
for ix in range(68):
    plt.plot(Xmats_ctrl[0, ix, :])


# In[ ]:





# ### Estimate Amats

# In[20]:


if not ("Amats_AD" in cur_res.keys()):
    Amats_ADs_lowrank = get_Amats(dXmats_AD[:], Xmats_AD[:], time_span, downrate=paras.downsample_rate, 
                              fct=paras.fct, nRks=paras.num_ranks, is_stack=False)
    cur_res.Amats_AD = Amats_ADs_lowrank
    save_pkl_dict2folder(RES_ROOT/_paras.folder_name, cur_res)


# In[21]:


if not ("Amats_ctrl" in cur_res.keys()):
    Amats_ctrls_lowrank = get_Amats(dXmats_ctrl[:], Xmats_ctrl[:], time_span, downrate=paras.downsample_rate, 
                              fct=paras.fct, nRks=paras.num_ranks, is_stack=False)
    cur_res.Amats_ctrl = Amats_ctrls_lowrank
    save_pkl_dict2folder(RES_ROOT/_paras.folder_name, cur_res)


# ### rank-R CP decomposition

# In[24]:


cur_res.keys()


# In[26]:


# transform to tensor
tensors_ctrl = np.transpose(np.array(cur_res.Amats_ctrl), (1, 2, 0))
tensors_AD = np.transpose(np.array(cur_res.Amats_AD), (1, 2, 0))

# the initial value 
# ctrl
Amat_ctrl_lowrank = np.sum(tensors_ctrl, axis=-1)
U, _, VT = np.linalg.svd(Amat_ctrl_lowrank)
ctrl_CPD_init = [U[:, :paras.r], VT.T[:, :paras.r]]

# AD
Amat_AD_lowrank = np.sum(tensors_AD, axis=-1)
U, _, VT = np.linalg.svd(Amat_AD_lowrank)
AD_CPD_init = [U[:, :paras.r], VT.T[:, :paras.r]]


# In[27]:


# rank-R decomposition
if not ("CPDresult_ctrl" in cur_res.keys()):
    CPDresult_ctrl = sort_orthCPD(decompose_three_way_orth(tensors_ctrl, paras.r, init=ctrl_CPD_init, verbose=True))
    cur_res.CPDresult_ctrl = CPDresult_ctrl
    save_pkl_dict2folder(_paras.save_dir, cur_res)


# In[52]:


# rank-R decomposition
if not ("CPDresult_AD" in cur_res.keys()):
    CPDresult_AD = sort_orthCPD(decompose_three_way_orth(tensors_AD, paras.r, init=AD_CPD_init, verbose=True))
    cur_res.CPDresult_AD = CPDresult_AD
    save_pkl_dict2folder(_paras.save_dir, cur_res)


# #### corrs with 7 networks

# In[53]:


tfn1 = partial(_cumsum_trunc, cutoff=0.90)
_corr_plot(cur_res.CPDresult_ctrl[0], cutoff=0.01, trun_fn=tfn1, trans_fn=lambda x:np.array(x));
_corr_plot(cur_res.CPDresult_AD[0], cutoff=0.01, trun_fn=tfn1, trans_fn=lambda x:x);


# In[54]:


tfn1 = partial(_cumsum_trunc, cutoff=0.90)
_corr_plot(np.abs(cur_res.CPDresult_ctrl[0]), cutoff=0.02, trun_fn=tfn1, trans_fn=lambda x:np.array(x));
_corr_plot(np.abs(cur_res.CPDresult_AD[0]), cutoff=0.02, trun_fn=tfn1, trans_fn=lambda x:x);


# In[ ]:





# ### Reduce the dim of the data

# In[61]:


nXmats_ctrl =  np.matmul(cur_res.CPDresult_ctrl[1].T[np.newaxis, :, :],
                         cur_res.dXXmats_ctrl[1])
nXmats_AD =  np.matmul(cur_res.CPDresult_AD[1].T[np.newaxis, :, :], 
                       cur_res.dXXmats_AD[1])

ndXmats_ctrl =  np.matmul(cur_res.CPDresult_ctrl[0].T[np.newaxis, :, :], 
                          cur_res.dXXmats_ctrl[0])
ndXmats_AD =  np.matmul(cur_res.CPDresult_AD[0].T[np.newaxis, :, :], 
                        cur_res.dXXmats_AD[0])


# In[62]:


if not ("nXmats_ctrl" in cur_res.keys()):
    cur_res.nXmats_ctrl = nXmats_ctrl
    save_pkl_dict2folder(_paras.save_dir, cur_res)
    
if not ("nXmats_AD" in cur_res.keys()):
    cur_res.nXmats_AD = nXmats_AD
    save_pkl_dict2folder(_paras.save_dir, cur_res)


# In[63]:


if not ("ndXmats_ctrl" in cur_res.keys()):
    cur_res.ndXmats_ctrl = ndXmats_ctrl
    save_pkl_dict2folder(_paras.save_dir, cur_res)
    
if not ("ndXmats_AD" in cur_res.keys()):
    cur_res.ndXmats_AD = ndXmats_AD
    save_pkl_dict2folder(_paras.save_dir, cur_res)


# In[ ]:





# ### Screening

# In[64]:


if not ("can_pts_ctrls" in cur_res.keys()):
    can_pts_ctrls = screening_4CPD(cur_res.ndXmats_ctrl, cur_res.nXmats_ctrl, wh=paras.wh)
    cur_res.can_pts_ctrls = can_pts_ctrls
    save_pkl_dict2folder(_paras.save_dir, cur_res)


# In[65]:


if not ("can_pts_ADs" in cur_res.keys()):
    can_pts_ADs = screening_4CPD(cur_res.ndXmats_AD, cur_res.nXmats_AD, wh=paras.wh)
    cur_res.can_pts_ADs = can_pts_ADs
    save_pkl_dict2folder(_paras.save_dir, cur_res)


# ### Detection

# In[66]:


if not ("cpts_ctrls" in cur_res.keys()):
    cpts_ctrls = []
    for ix in trange(len(cur_res.can_pts_ctrls)):
        res = dyna_prog_4CPD(cur_res.ndXmats_ctrl[ix], 
                             cur_res.nXmats_ctrl[ix], 
                             paras.kappa, 
                             Lmin=paras.Lmin,  
                             canpts=cur_res.can_pts_ctrls[ix], 
                             maxM=paras.maxM,  
                             is_full=True,  
                             showProgress=False)
        cpts_ctrls.append(res)
    
    cur_res.cpts_ctrls = cpts_ctrls
    save_pkl_dict2folder(_paras.save_dir, cur_res)


# In[67]:


if not ("cpts_ADs" in cur_res.keys()):
    cpts_ADs = []
    for ix in trange(len(cur_res.can_pts_ADs)):
        res = dyna_prog_4CPD(cur_res.ndXmats_AD[ix], 
                             cur_res.nXmats_AD[ix], 
                             paras.kappa, 
                             Lmin=paras.Lmin,  
                             canpts=cur_res.can_pts_ADs[ix], 
                             maxM=paras.maxM,  
                             is_full=True,  
                             showProgress=False)
        cpts_ADs.append(res)
    
    cur_res.cpts_ADs = cpts_ADs
    save_pkl_dict2folder(_paras.save_dir, cur_res)


# In[ ]:




