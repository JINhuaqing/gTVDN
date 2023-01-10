#!/usr/bin/env python
# coding: utf-8

# This file analyze the results for regression.

# In[1]:


import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

from pathlib import Path
from pprint import pprint
from scipy.io import loadmat
from tqdm import trange, tqdm
from scipy.stats import ttest_ind
from collections import defaultdict as ddict
from easydict import EasyDict as edict

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


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


# In[3]:


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

import utils.brain_plot
importlib.reload(utils.brain_plot)
from utils.brain_plot import reorder_U, U_2brain_vec


# In[ ]:





# ## Parameters

# In[4]:


pprint(paras)


# In[5]:


# this parameters only for this file
_paras = edict()
_paras.folder_name = "method_eig"
_paras.save_dir = RES_ROOT/_paras.folder_name
print(f"Load results to {_paras.save_dir}")


# ## Load results

# In[17]:


# load results
cur_res = load_pkl_folder2dict(_paras.save_dir)


# In[7]:


pprint(cur_res.keys())


# ## get the dataset

# In[8]:


ncpts_ADs_selected = [len(cpts) for cpts in cur_res.cpts_ADs_selected]
ncpts_ctrls_selected = [len(cpts) for cpts in cur_res.cpts_ctrls_selected]


# In[135]:


# selected U and calculate weighted U (ABS wU)
wUs_abs_AD = []
wUs_abs_mean_AD = []
wUs_abs_max_AD = []
ws_abs_mean_AD = []
ws_abs_max_AD = []
for ix in range(len(cur_res.eig_vals_ADs)):
    cur_eig_val_abs = np.abs(cur_res.eig_vals_ADs[ix])
    cur_dwells = cur_res.dwells_ADs_selected[ix]
    
    cur_ws_abs = cur_eig_val_abs
    cur_Us_abs = np.abs(cur_res.dim_redu_ctrl.eigVecs[:, :cur_res.post_paras.ctrl_r])
    
    cur_wUs_abs = np.matmul(cur_Us_abs, cur_ws_abs)
    cur_wUs_abs_mean = cur_wUs_abs.mean(axis=1)
    
    wUs_abs_AD.append(cur_wUs_abs)
    wUs_abs_max_AD.append(cur_wUs_abs[:, np.argmax(cur_dwells)])
    wUs_abs_mean_AD.append(cur_wUs_abs_mean)
    ws_abs_mean_AD.append(cur_eig_val_abs[:, :].mean())
    ws_abs_max_AD.append(cur_eig_val_abs[:, :].max())
    
wUs_abs_ctrl = []
wUs_abs_mean_ctrl = []
wUs_abs_max_ctrl = []
ws_abs_mean_ctrl = []
ws_abs_max_ctrl = []
for ix in range(len(cur_res.eig_vals_ctrls)):
    cur_eig_val_abs = np.abs(cur_res.eig_vals_ctrls[ix])
    cur_dwells = cur_res.dwells_ctrls_selected[ix]
    
    cur_ws_abs = cur_eig_val_abs
    cur_Us_abs = np.abs(cur_res.dim_redu_AD.eigVecs[:, :cur_res.post_paras.AD_r])
    
    cur_wUs_abs = np.matmul(cur_Us_abs, cur_ws_abs)
    cur_wUs_abs_mean = cur_wUs_abs.mean(axis=1)
    
    wUs_abs_ctrl.append(cur_wUs_abs)
    wUs_abs_max_ctrl.append(cur_wUs_abs[:, np.argmax(cur_dwells)])
    wUs_abs_mean_ctrl.append(cur_wUs_abs_mean)
    ws_abs_mean_ctrl.append(cur_eig_val_abs[:, :].mean())
    ws_abs_max_ctrl.append(cur_eig_val_abs[:, :].max())
    
wUs_abs_mean_AD = np.array(wUs_abs_mean_AD)
wUs_abs_mean_ctrl = np.array(wUs_abs_mean_ctrl)
wUs_abs_max_AD = np.array(wUs_abs_max_AD)
wUs_abs_max_ctrl = np.array(wUs_abs_max_ctrl)


# ## Regression 

# In[136]:


import numbers
# return the predicted probs for each test obs
def clf_2probs(clf, X_test):
    probs = clf.predict_proba(X_test)
    return probs[:, clf.classes_==1].reshape(-1)

def LOO_pred_givenC(cur_X, cur_Y, Cs=1, is_prg=True):
    probs = []
    if is_prg:
        prog_bar = trange(len(cur_Y))
    else:
        prog_bar = range(len(cur_Y))
    if isinstance(Cs, numbers.Number):
        Cs = np.ones_like(cur_Y)*Cs
    for ix in prog_bar:
        cur_X_test = cur_X[ix, :].reshape(1, -1)
        cur_Y_test = cur_Y[ix].reshape(1, -1)
        cur_X_train = np.delete(cur_X, ix, axis=0)
        cur_Y_train = np.delete(cur_Y, ix)
        clf = LogisticRegression(random_state=0, C=Cs[ix], penalty="l2", solver="liblinear").fit(cur_X_train, cur_Y_train)
        #clf = RandomForestClassifier(random_state=0).fit(cur_X_train, cur_Y_train)
        #clf = DecisionTreeClassifier(random_state=0).fit(cur_X_train, cur_Y_train)
        probs.append(clf_2probs(clf, cur_X_test))
    return np.array(probs).reshape(-1)


# In[137]:


def LOO_bestC(cur_X, cur_Y, Cs, is_prg=0, is_C_only=1):
    Cs = np.array(Cs)
    cur_aucs = []
    if is_prg:
        prog_bar = tqdm(Cs)
    else:
        prog_bar = Cs
    for cur_C in prog_bar: 
        cur_pred_probs = LOO_pred_givenC(cur_X, cur_Y, Cs=cur_C, is_prg=0)
        cur_auc = roc_auc_score(cur_Y, cur_pred_probs)
        cur_aucs.append(cur_auc)
        cur_best_C = Cs[np.argmax(cur_aucs)]
    cur_best_C = Cs[np.argmax(cur_aucs)]
    if is_C_only:
        return cur_best_C
    else:
        return cur_best_C, cur_aucs


# In[138]:


def tmp_foldwU(mat):
    assert mat.shape[-1] == 68
    mat_half = (mat[:, :34] + mat[:, 34:])/2
    return mat_half


# In[139]:


cur_res.keys()


# In[140]:


# prepare for Y and X
reg_Y = np.concatenate([np.ones_like(ncpts_ADs_selected),
                        np.zeros_like(ncpts_ctrls_selected)])
reg_X_wU_abs_mean = np.concatenate([wUs_abs_mean_AD, wUs_abs_mean_ctrl], axis=0)
reg_X_wU_abs_max = np.concatenate([wUs_abs_max_AD, wUs_abs_max_ctrl], axis=0)
##reg_X_rank = np.concatenate([cur_res.rank_ADs, cur_res.rank_ctrls])
reg_X_ncpts = np.concatenate([ncpts_ADs_selected, ncpts_ctrls_selected])
reg_X_ws_max = np.concatenate([ws_abs_max_AD, ws_abs_max_ctrl])
reg_X_ws_mean = np.concatenate([ws_abs_mean_AD, ws_abs_mean_ctrl])


reg_X = np.concatenate([
                        #tmp_foldwU(reg_X_wU_abs_mean), # try minmax each row
                        reg_X_wU_abs_mean, 
                        reg_X_ncpts.reshape(-1, 1), 
                        reg_X_ws_max.reshape(-1, 1)
                       ], axis=1)
##reg_X = reg_X_wU_abs_max
reg_X_std = (reg_X - reg_X.mean(axis=0))/reg_X.std(axis=0)


# In[ ]:





# ### AUC under LOO

# In[141]:


gopt_C = LOO_bestC(reg_X_std, reg_Y, paras.Cs, 1)
print(gopt_C, paras.Cs)


# In[142]:


pred_probs = LOO_pred_givenC(reg_X_std, reg_Y, Cs=gopt_C, is_prg=1)
fpr, tpr, thresholds = roc_curve(reg_Y, pred_probs, pos_label=1)
auc = roc_auc_score(reg_Y, pred_probs)

plt.title(f"ROC (AUC:{auc:.3f})")
plt.plot(fpr, tpr)


# In[ ]:





# ### CV for tuning C for each Obs

# In[33]:


import multiprocessing as mp
def run_fn(ix):
    cur_X_test = reg_X_std[ix, :].reshape(1, -1)
    cur_Y_test = reg_Y[ix].reshape(1, -1)
    cur_X_train = np.delete(reg_X_std, ix, axis=0)
    cur_Y_train = np.delete(reg_Y, ix)
    
    print(f"Start {ix}")
    cur_best_C =  LOO_bestC(cur_X_train, cur_Y_train, paras.Cs, 0)
    print(f"Finished {ix}")
    return (ix, cur_best_C)


if __name__ == "__main__":
    with mp.Pool(processes=20) as pool:
        res_proc = []
        for ix in range(len(reg_Y)):
            res_proc.append( pool.apply_async(run_fn, [ix,]) )
        res = [cur_proc.get() for cur_proc in res_proc] # to retrieve the results
    pool.join()


# In[34]:


res = sorted(res, key=lambda x:x[0])
best_Cs = [re[-1] for re in res]

probs = []
pred_probs = LOO_pred_givenC(reg_X_std, reg_Y, Cs=best_Cs, is_prg=1)
fpr, tpr, thresholds = roc_curve(reg_Y, pred_probs, pos_label=1)
auc = roc_auc_score(reg_Y, pred_probs)

plt.title(f"ROC (AUC:{auc:.3f})")
plt.plot(fpr, tpr)


# ### AUC under 10000 CV

# In[61]:


# AUC under repetitions
np.random.seed(0)
nobs = reg_X_std.shape[0]
rep_aucs = []
for j in tqdm(range(10000)):
    test_idxs = np.random.choice(nobs, int(nobs/5), False)
    train_idxs = np.delete(np.arange(nobs), test_idxs)
    clf = LogisticRegression(penalty=paras.penalty, random_state=0, C=gopt_C)
    clf.fit(reg_X_std[train_idxs], reg_Y[train_idxs])
    cur_eprobs = clf_2probs(clf, reg_X_std[test_idxs, :])
    cur_auc = roc_auc_score(reg_Y[test_idxs], cur_eprobs)
    rep_aucs.append(cur_auc)
mean_auc = np.mean(rep_aucs)
std_auc = np.std(rep_aucs)
print(f"The mean of AUC under 1000 repetitions is {mean_auc:.3f} and the standard deviation is {std_auc:.3f}, "
      f"the 95% CI is ({np.quantile(rep_aucs, 0.025):.3f}, {np.quantile(rep_aucs, 0.975):.3f}).")


# ### final fit  and bootstrap analysis

# In[76]:


final_clf = LogisticRegression(random_state=0, C=gopt_C, 
                               penalty="l2").fit(reg_X_std, reg_Y)
                               #solver="liblinear").fit(reg_X_std, reg_Y)
final_coefs = final_clf.coef_.reshape(-1)


# In[72]:


# bootstrap CIs
np.random.seed(1)
rep_num = 10000
parass_boot = []
for _ in trange(rep_num):
    boot_idx = np.random.choice(len(reg_Y), len(reg_Y))
    cur_Y_boot = reg_Y[boot_idx]
    cur_X_boot = reg_X_std[boot_idx]
    cur_clf = LogisticRegression(penalty=paras.penalty, random_state=0, C=gopt_C)
    cur_clf.fit(cur_X_boot, cur_Y_boot)
    paras_boot = cur_clf.coef_.reshape(-1)
    parass_boot.append(paras_boot)


# In[64]:


parass_boot = np.array(parass_boot)
# 95% CIs
lows, ups = final_coefs-parass_boot.std(axis=0)*1.96, final_coefs+parass_boot.std(axis=0)*1.96
keep_idx_boot = np.bitwise_or(lows >0,  ups < 0)
keep_idx_id_boot =  np.where(keep_idx_boot)[0]

# Pvalue
test_stat_boot = final_coefs/parass_boot.std(axis=0)
norm_rv = scipy.stats.norm()
# to be consistent, I think we should use two-sided pvalue
boot_pvs = 2*(1-norm_rv.cdf(np.abs(test_stat_boot)))


# In[65]:


xlabs = np.concatenate([REGION_NAMES, ["Num of cpts"]])
plt.figure(figsize=[15, 4])
plt.fill_between(list(range(len(lows))), lows, ups, color="red", alpha=.5)
plt.title("The 95% CI of the parameters")
_ = plt.xticks(keep_idx_id_boot, xlabs[keep_idx_id_boot], rotation=30, fontsize=8)


# In[66]:


# Bootstrap p value and CI are consistent
tmp_idx = np.bitwise_xor(boot_pvs <= 0.05, keep_idx_boot)
res_tb = {
   "Feature" : xlabs[tmp_idx],  
    "Parameters": final_coefs[tmp_idx],
    "Lower": lows[tmp_idx],
    "Upper": ups[tmp_idx],
    "Pvalues": boot_pvs[tmp_idx]
}
pd.set_option("display.precision", 3)
res_tb = pd.DataFrame(res_tb)
print(res_tb)


# In[67]:


#  output table
res_tb = {
   "Feature" : xlabs[keep_idx_boot],  
    "Parameters": final_coefs[keep_idx_boot],
    "Lower": lows[keep_idx_boot],
    "Upper": ups[keep_idx_boot],
    "Pvalues": boot_pvs[keep_idx_boot]
}
#pd.set_option("display.precision", 3)
pd.set_option('display.float_format',lambda x : '%.4f' % x)
res_tb = pd.DataFrame(res_tb)
print(res_tb)


# In[200]:





# In[201]:


# final parameter, no abs, remove pv <0.05
mm_pn_paras = minmax(np.abs(np.concatenate([final_coefs[:34], final_coefs[:34]])))
nlog_pvs = -np.log10(np.concatenate([boot_pvs[:34], boot_pvs[:34]]))
mm_pn_paras[nlog_pvs < -np.log10(0.05)] = 0
out_paras = U_2brain_vec(reorder_U(mm_pn_paras))
print(np.sum(mm_pn_paras ==0), np.sum(nlog_pvs < -np.log10(0.05)))
np.savetxt(RES_ROOT/f"./{_paras.folder_name}/abs_paras_part.txt", out_paras)
#pd.DataFrame({"Name": REGION_NAMES, 
#              "Vec": mm_pn_paras}).to_csv(RES_ROOT/f"./{_paras.folder_name}/abs_paras.csv", 
#                                          index=False)


# In[202]:


mm_pn_paras


# In[197]:


# final parameter, no abs, remove pv <0.05
mm_pn_paras = minmax(np.abs(final_coefs[:68]))
nlog_pvs = -np.log10(boot_pvs[:68])
mm_pn_paras[nlog_pvs < -np.log10(0.05)] = 0
out_paras = U_2brain_vec(reorder_U(mm_pn_paras))
print(np.sum(mm_pn_paras ==0), np.sum(nlog_pvs < -np.log10(0.05)))
np.savetxt(RES_ROOT/f"./{_paras.folder_name}/abs_paras.txt", out_paras)
#pd.DataFrame({"Name": REGION_NAMES, 
#              "Vec": mm_pn_paras}).to_csv(RES_ROOT/f"./{_paras.folder_name}/abs_paras.csv", 
#                                          index=False)


# In[ ]:




