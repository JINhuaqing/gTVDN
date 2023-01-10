import numpy as np
from pathlib import Path

_cur_dir = Path(__file__).parent

with open(_cur_dir/"../../data/BNVtemplate_DK68.txt", "r") as tf:
    _DKtmplateRaw = tf.readlines()
_DKtmplate = np.array([int(x.strip()) for x in _DKtmplateRaw])


def U_2brain_vec(wU):
    emVec = np.zeros_like(_DKtmplate, dtype=np.float64)
    for idx in range(1, 69):
        emVec[_DKtmplate==idx] = wU[idx-1]
    return emVec

# reorder U from left-first-then-right to left-to-right-alternative
def reorder_U(wU):
    wUreorder = np.zeros_like(wU, dtype=np.float64)
    wUreorder[0::2] = wU[:34]
    wUreorder[1::2] = wU[34:]
    return wUreorder


# obt FC matrix with exp
def obt_FC_exp(vec, remove_diag=True, normalize=-1):
    if normalize is None:
        normalize = 1
    if normalize < 0:
        normalize = np.std(vec)
    FC = np.exp(-np.abs(vec.reshape(-1, 1) - vec.reshape(1, -1))/normalize)
    if remove_diag:
        FC = FC - np.diag(np.diag(FC))
    return FC 