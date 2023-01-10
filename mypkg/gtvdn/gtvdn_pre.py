# this file contains fns related to preprocessing of gTVDN
from scipy.signal import detrend, decimate

def preprocess_MEG(Ymats, paras):
    """args:
            Ymats: multiple MEG datasets, N x d x n, 
                   N is num of datasets, d is number of ROIs and n is the length of the sequences
            paras: Parameters for preprocess, dict
                   is_detrend: whether or detrend or not, Bool
                   decimate_rate: The rate to decimate from MEG data, reduce the resolution. If None, not decimate. Integer or None
    """
    # Decimate the data first
    decimate_rate = paras.decimate_rate
    if decimate_rate is not None:
        Ymats = decimate(Ymats, decimate_rate, ftype="fir")

    # Then Detrend the data
    is_detrend = paras.is_detrend
    if is_detrend:
        Ymats = detrend(Ymats)
    return Ymats