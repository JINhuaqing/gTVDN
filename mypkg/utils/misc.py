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

# load file from folder and save it to dict
def load_pkl_folder2dict(folder, vars_=None):
    res = edict()
    for fil in folder.glob('*.pkl'):
        if vars_ is None:
            res[fil.stem] = load_pkl(fil)
        else:
            if fil.stem in vars_:
                res[fil.stem] = load_pkl(fil)
    return res

# save a dict into a folder
def save_pkl_dict2folder(folder, res, is_force=False):
    assert isinstance(res, dict)
    for ky, v in res.items():
        save_pkl(folder/f"{ky}.pkl", v, is_force=is_force)

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
def cumsum_cutoff(vec, cutoff=0.8):
    vec = np.abs(vec)
    sorted_idx = np.argsort(-vec)
    ratio_vec = np.cumsum(vec[sorted_idx])/np.sum(vec)
    rank = np.sum(ratio_vec <cutoff) + 1
    keep_idx = np.sort(sorted_idx[:rank])
    return keep_idx


# load file from pkl
def load_pkl(fil):
    print(f"Load file {fil}")
    with open(fil, "rb") as f:
        result = pickle.load(f)
    return result

# save file to pkl
def save_pkl(fil, result, is_force=False):
    if not fil.parent.exists():
        fil.parent.mkdir()
        print(fil.parent)
        print(f"Create a folder {fil.parent}")
    if is_force or (not fil.exists()):
        print(f"Save to {fil}")
        with open(fil, "wb") as f:
            pickle.dump(result, f)
    else:
        print(f"{fil} exists! Use is_force=True to save it anyway")