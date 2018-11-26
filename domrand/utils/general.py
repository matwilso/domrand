import numpy as np
import tensorflow as tf
from domrand.utils.constants import TBS, TBE, TBS_TF, TBE_TF

# TODO: write docs and tests for these
# TODO: add softmax version of binpreds
# TODO: add more checking

def bin_to_xyz_np(arr, bins):
    """Convert xyz coords into a number of bins on the table"""
    bindiv = bins - 1.0
    xyz = ((arr / bindiv) * (TBE - TBS)) + TBS
    return xyz

def bin_to_xyz_tf(arr, bins):
    """Same as bin_to_xyz_np, but using TensorFlow ops"""
    bindiv = tf.cast(bins, tf.float32) - 1.0
    arr = tf.cast(arr, tf.float32)
    frac = (arr / bindiv)
    xyz = (frac * (TBE_TF - TBS_TF)) + TBS_TF
    return xyz
    
def notify(message):
    """Send a message with linux notify-send"""
    import subprocess
    subprocess.call(['notify-send', message])

def softmax(X, temperature=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats. 
    temperature (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the 
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.

    Source: https://nolanbconaway.github.io/blog/2017/softmax-numpy 
    """
    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the temperature parameter, 
    y = y * float(temperature)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p