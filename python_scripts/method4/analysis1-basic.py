#!/usr/bin/env python
# coding: utf-8
This file analysis the result from `method3`

1. select kappa and ranks, some simple plots of them

2. Reestimate the singular values for each segment based on the selected kappa
# In[17]:


import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

from pathlib import Path
from scipy.io import loadmat
from pprint import pprint
from tqdm import trange, tqdm
from scipy.stats import ttest_ind
from collections import defaultdict as ddict
from easydict import EasyDict as edict

get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


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
from constants import REGION_NAMES, REGION_NAMES_WLOBE, DATA_ROOT, RES_ROOT, FIG_ROOT


# In[9]:


# utils
import utils.matrix
importlib.reload(utils.matrix)
from utils.matrix import eig_sorted

import utils.misc
importlib.reload(utils.misc)
from utils.misc import paras2name, cumsum_cutoff, load_pkl_folder2dict, save_pkl_dict2folder

import utils.standardize
importlib.reload(utils.standardize)
from utils.standardize import minmax, minmax_mat, minmax_pn


# In[5]:


# gtvdn
import gtvdn.gtvdn_post
importlib.reload(gtvdn.gtvdn_post)
from gtvdn.gtvdn_post import est_singular_vals, update_kp


# ## Parameters

# In[18]:


pprint(paras)


# In[7]:


# this parameters only for this file
_paras = edict()
_paras.folder_name = "method3"


# In[32]:


# Lets save some results
post_paras = edict()


# In[ ]:





# ## Load results

# In[41]:


# load results
cur_res = load_pkl_folder2dict(RES_ROOT/_paras.folder_name)


# In[42]:


CPDresult_ctrl = cur_res.CPDresult_ctrl
CPDresult_AD = cur_res.CPDresult_AD
cpts_ctrls = cur_res.cpts_ctrls
cpts_ADs = cur_res.cpts_ADs
ndXmats_AD = cur_res.ndXmats_AD
ndXmats_ctrl = cur_res.ndXmats_ctrl
nXmats_AD = cur_res.nXmats_AD
nXmats_ctrl = cur_res.nXmats_ctrl



B1 = CPDresult_ctrl[0]
B2 = CPDresult_ctrl[1]


# In[ ]:





# ##  Curves of singular values

# In[12]:


B3_abs_AD = np.abs(CPDresult_AD[-1])
B3_abs_ctrl = np.abs(CPDresult_ctrl[-1])


# In[13]:


B3_abs_mm_AD = minmax_mat(B3_abs_AD, is_row=True)
B3_abs_mm_ctrl = minmax_mat(B3_abs_ctrl, is_row=True)
for ix in range(88):
    plt.plot(B3_abs_mm_AD[ix, :], "-b")
for ix in range(88):
    plt.plot(B3_abs_mm_ctrl[ix, :], "-r")


# In[14]:


AD_mean = np.mean(B3_abs_mm_AD[:, :], axis=0)
AD_std = B3_abs_mm_AD.std(axis=0)
AD_low, AD_up = AD_mean - 1.96*AD_std/np.sqrt(88), AD_mean + 1.96*AD_std/np.sqrt(88)

ctrl_mean = np.mean(B3_abs_mm_ctrl[:, :], axis=0)
ctrl_std = B3_abs_mm_ctrl.std(axis=0)
ctrl_low, ctrl_up = ctrl_mean - 1.96*ctrl_std/np.sqrt(88), ctrl_mean + 1.96*ctrl_std/np.sqrt(88)


# In[15]:


plt.plot(AD_mean, label="AD")
plt.plot(ctrl_mean, label="Ctrl")
plt.fill_between(np.arange(20), AD_low, AD_up, alpha=0.3)
plt.fill_between(np.arange(20), ctrl_low, ctrl_up, alpha=0.3)
plt.legend()


# ## Select kappa

# In[30]:


# sel kappa
cpts_ctrls_allkp = []
_, cur_r, n = ndXmats_ctrl.shape
for kp in paras.kps:
    cpts_ctrl_allkp = np.array([update_kp(kp, ix.U0, n, cur_r, paras) for ix in cpts_ctrls])
    cpts_ctrls_allkp.append(cpts_ctrl_allkp)
    

cpts_ADs_allkp = []
_, cur_r, n = ndXmats_AD.shape
for kp in paras.kps:
    cpts_AD_allkp = np.array([update_kp(kp, ix.U0, n, cur_r, paras) for ix in cpts_ADs])
    cpts_ADs_allkp.append(cpts_AD_allkp)


# In[33]:


cpts_ctrls_allkp = np.array(cpts_ctrls_allkp)
cpts_ADs_allkp = np.array(cpts_ADs_allkp)
cpts_ctrl_std = cpts_ctrls_allkp.std(axis=1)
cpts_ctrl_mean = cpts_ctrls_allkp.mean(axis=1)

kps_keepidx_ctrl = np.bitwise_and(cpts_ctrls_allkp.mean(axis=1)>=3, cpts_ctrls_allkp.mean(axis=1)<=18)
#kps_keepidx_AD = np.bitwise_and(cpts_ADs_allkp.mean(axis=1)>=2, cpts_ADs_allkp.mean(axis=1)<=18)
#kps_keepidx_2grp = np.bitwise_and(kps_keepidx_AD, kps_keepidx_ctrl)


plt.plot(paras.kps[kps_keepidx_ctrl], cpts_ctrl_std[kps_keepidx_ctrl])
plt.xlabel("Kappa")
plt.ylabel("Std")


sel_kp = paras.kps[kps_keepidx_ctrl][np.argmin(cpts_ctrl_std[kps_keepidx_ctrl])]
post_paras.sel_kp = sel_kp
sel_kp


# ### change pts under selected kappa

# In[34]:


cur_kp = sel_kp
_, cur_r, n = ndXmats_ctrl.shape
ncpts_ctrls_selected = np.array([update_kp(cur_kp, ix.U0, n, cur_r, paras) 
                                 for ix in cpts_ctrls])
cpts_ctrls_selected = [cur_cpts.chgMat[cur_ncpt-1, :cur_ncpt] 
                       for cur_ncpt, cur_cpts 
                       in zip(ncpts_ctrls_selected, cpts_ctrls)]


_, cur_r, n = ndXmats_AD.shape
ncpts_ADs_selected = np.array([update_kp(cur_kp, ix.U0, n, cur_r, paras) 
                               for ix 
                               in cpts_ADs])
cpts_ADs_selected = [cur_cpts.chgMat[cur_ncpt-1, :cur_ncpt] 
                     for cur_ncpt, cur_cpts 
                     in zip(ncpts_ADs_selected, cpts_ADs)]


# In[47]:


dwells_ADs_selected = [np.diff(np.concatenate([[1], cpts_selected, [7200]]))/120
                       for cpts_selected 
                       in cpts_ADs_selected]
dwells_ctrls_selected = [np.diff(np.concatenate([[1], cpts_selected, [7200]]))/120
                         for cpts_selected 
                         in cpts_ctrls_selected]


# In[36]:


pval = ttest_ind(ncpts_ADs_selected, ncpts_ctrls_selected, equal_var=False).pvalue

plt.figure(figsize=[10, 5])
plt.subplot(121)
plt.hist(ncpts_ADs_selected, alpha=0.5, label="AD")
plt.hist(ncpts_ctrls_selected, alpha=0.5, label="Ctrl")
plt.title(f"T-test Pvalue {pval:.3E}")
plt.legend()

plt.subplot(122)
sns.boxplot(data=[ncpts_ADs_selected, ncpts_ctrls_selected], showfliers=0)
plt.xticks([0, 1], ["AD", "Ctrl"]);
plt.title(f"AD ({ncpts_ADs_selected.mean():.3f}) vs Ctrl ({ncpts_ctrls_selected.mean():.3f})")


# ##  Rank results

# In[37]:


paras.cutoff=0.6
post_paras.rank_curoff = paras.cutoff 
#rank_ADs = np.sum(minmax_mat(np.abs(CPDresult_AD[2]), is_row=True) > paras.cutoff, axis=1)
#rank_ctrls = np.sum(minmax_mat(np.abs(CPDresult_ctrl[2]), is_row=True) > paras.cutoff, axis=1)
rank_ADs = np.array([len(cumsum_cutoff(np.abs(CPDresult_AD[2][ix]), paras.cutoff)) 
                     for ix 
                     in range(len(CPDresult_AD[2]))])
rank_ctrls = np.array([len(cumsum_cutoff(np.abs(CPDresult_ctrl[2][ix]), paras.cutoff)) 
                       for ix 
                       in range(len(CPDresult_ctrl[2]))])

pval = ttest_ind(rank_ADs, rank_ctrls, equal_var=False).pvalue

plt.figure(figsize=[10, 5])
plt.subplot(121)
plt.hist(rank_ADs, alpha=0.5, label="AD")
plt.hist(rank_ctrls, alpha=0.5, label="Ctrl")
plt.title(f"T-test Pvalue {pval:.3E}")
plt.legend()

plt.subplot(122)
sns.boxplot(data=[rank_ADs, rank_ctrls], showfliers=0)
plt.xticks([0, 1], ["AD", "Ctrl"]);
plt.title(f"AD ({rank_ADs.mean():.3f}) vs Ctrl ({rank_ctrls.mean():.3f})")


# ## Re-estimate the lambdas for each segment

# In[43]:


singular_vals_ADs = []
singular_vals_ctrls = []

for ix in range(len(cpts_ADs_selected)):
    cur_singular_vals = est_singular_vals(cpts_ADs_selected[ix], ndXmats_AD[ix], nXmats_AD[ix])
    singular_vals_ADs.append(cur_singular_vals)
    
for ix in range(len(cpts_ctrls_selected)):
    cur_singular_vals = est_singular_vals(cpts_ctrls_selected[ix], ndXmats_ctrl[ix], nXmats_ctrl[ix])
    singular_vals_ctrls.append(cur_singular_vals)


# ## Save updated results

# In[48]:


cur_res.post_paras = post_paras

cur_res.singular_vals_ADs = singular_vals_ADs
cur_res.cpts_ADs_selected = cpts_ADs_selected
cur_res.dwells_ADs_selected = dwells_ADs_selected
cur_res.rank_ADs = rank_ADs

cur_res.singular_vals_ctrls = singular_vals_ctrls
cur_res.cpts_ctrls_selected = cpts_ctrls_selected
cur_res.dwells_ctrls_selected = dwells_ctrls_selected
cur_res.rank_ctrls = rank_ctrls


# In[49]:


save_pkl_dict2folder(RES_ROOT/_paras.folder_name, cur_res, True)


# In[ ]:




