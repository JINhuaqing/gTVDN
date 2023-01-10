import numpy as np

# minmax a matrix between [0, 1], byrow or bycol
def minmax_mat(mat, is_row=True):
    if is_row:
        mins = np.min(mat, axis=1)[:, np.newaxis]
        maxs = np.max(mat, axis=1)[:, np.newaxis]
    else:
        mins = np.min(mat, axis=0)[np.newaxis, :]
        maxs = np.max(mat, axis=0)[np.newaxis, :]
    rv = (mat-mins)/(maxs-mins)
    return rv


# minmax a vec between [-1, 1]
def minmax_pn(x):
    num = 2 * (x-np.min(x))
    den = np.max(x) - np.min(x)
    return num/den - 1


# minmax a vec between [0, 1]
def minmax(x):
    rev = (x - np.min(x))/(np.max(x) - np.min(x))
    return rev

# minmax a vec between [dlt, 1], dlt is a small number > 0
# for brainplotting, 0 will be treated as NULL.
def minmax_plotting(vec):
    min_dlt = np.min(vec)-0.01
    max_v = np.max(vec)
    res = (vec-min_dlt)/(max_v-min_dlt)
    return res