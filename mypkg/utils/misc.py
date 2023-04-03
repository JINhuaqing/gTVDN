import numpy as np
import scipy
import pickle
from easydict import EasyDict as edict
from sklearn.manifold import SpectralEmbedding


# reduce the dim with Spectral Embedding
def spec_emb_red(Ys, n_out=1):
    """Ys: num of obs x num of dim
    """
    embedding = SpectralEmbedding(n_components=n_out)
    Ys_transformed = embedding.fit_transform(Ys)
    return Ys_transformed

def load_pkl_folder2dict(folder, excluding=[], including=["*"], verbose=True):
    """The function is to load pkl file in folder as an edict
        args:
            folder: the target folder
            excluding: The files excluded from loading
            including: The files included for loading
            Note that excluding override including
    """
    if not isinstance(including, list):
        including = [including]
    if not isinstance(excluding, list):
        excluding = [excluding]
        
    if len(including) == 0:
        inc_fs = []
    else:
        inc_fs = list(set(np.concatenate([list(folder.glob(nam+".pkl")) for nam in including])))
    if len(excluding) == 0:
        exc_fs = []
    else:
        exc_fs = list(set(np.concatenate([list(folder.glob(nam+".pkl")) for nam in excluding])))
    load_fs = np.setdiff1d(inc_fs, exc_fs)
    res = edict()
    for fil in load_fs:
        res[fil.stem] = load_pkl(fil, verbose)                                                                                                                                  
    return res

# save a dict into a folder
def save_pkl_dict2folder(folder, res, is_force=False, verbose=True):
    assert isinstance(res, dict)
    for ky, v in res.items():
        save_pkl(folder/f"{ky}.pkl", v, is_force=is_force, verbose=verbose)

def paras2name(paras, ext=".pkl"):
    names = []
    for k, v in paras.items():
        if isinstance(v, bool) or isinstance(v, int):
            names.append(f"{k}{v}__")
        elif isinstance(v, float):
            names.append(f"{k}{v:.2E}__")
    name = "".join(names)
    return name[:-2] + ext





# return the idxs to keep based on the cumsum cumulated cutoff
def cumsum_cutoff(vec, cutoff=0.8, ord_=1):
    """This fn is to select rank based on the L-ord_ norm cutoff
    """
    vec = np.abs(vec)
    sorted_idx = np.argsort(-vec)
    ratio_vec = np.cumsum(vec[sorted_idx]**ord_)/(np.linalg.norm(vec, ord=ord_)**ord_)
    rank = np.sum(ratio_vec <cutoff) + 1
    keep_idx = np.sort(sorted_idx[:rank])
    return keep_idx


# load file from pkl
def load_pkl(fil, verbose=True):
    if verbose:
        print(f"Load file {fil}")
    with open(fil, "rb") as f:
        result = pickle.load(f)
    return result

# save file to pkl
def save_pkl(fil, result, is_force=False, verbose=True):
    if not fil.parent.exists():
        fil.parent.mkdir()
        if verbose:
            print(fil.parent)
            print(f"Create a folder {fil.parent}")
    if is_force or (not fil.exists()):
        if verbose:
            print(f"Save to {fil}")
        with open(fil, "wb") as f:
            pickle.dump(result, f)
    else:
        if verbose:
            print(f"{fil} exists! Use is_force=True to save it anyway")
        else:
            pass