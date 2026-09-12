"""
@author: marcelo looser
"""

import numpy as np

# =============================================================================

def mutual_coherence(a):
    """
    Calculates the mutual coherence of a matrix.

    Parameters
    ----------
    a : array_like
        Input matrix.

    Returns
    -------
    Mutual coherence.
    """
    gram, _ = _gram_matrix(a=a, normed=True)
    gram.ravel()[::gram.shape[1]+1] = 0 # setting diagonal elements to zero
    return abs(gram).max()


def cross_coherence(a, b, norm=2):
    """
    Calculates the cross coherence between two matrices.

    Parameters
    ----------
    a : array_like
        First input matrix of shape (k, m).
    b : array_like
        Second input matrix of shape (m, n).
    norm : int, optional
        Norm to use for normalization.

    Returns
    -------
    Cross coherence.
    """
    return abs(((a/np.linalg.norm(a, ord=norm, axis=0))).conj().dot((b/np.linalg.norm(b, ord=norm, axis=0)))).max()


def t_average_coherence(a, t=0):
    """
    Calculates the t-average coherence of a matrix.
    
    Parameters
    ----------
    a : array_like
        Input matrix.
    t : float, optional
        Threshold for coherence calculation. Default is 0.
    
    Returns
    -------
    t-average coherence.
    """
    gram, _ = _gram_matrix(a=a, normed=False)
    gram.ravel()[::gram.shape[1]+1] = 0 # setting diagonal elements to zero
    return np.mean(abs(gram)*(abs(gram)) > t)


def _gram_matrix(a, normed=False):
    """
    Constructs the Gram matrix of a given input matrix and its conjugate transpose.

    Parameters
    ----------
    a : array_like
        Input matrix.
    normed : bool, optional
        Whether to normalize the columns of the input matrix.

    Returns
    -------
    Gram matrix and the conjugate transpose of a.
    """
    if normed:
        a = a/np.linalg.norm(a, ord=2, axis=0)
    ah = a.conj().T
    gram = ah.dot(a)
    return gram, ah

# =============================================================================
