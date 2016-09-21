import numpy
import theano.tensor as T
from . import cost 

def orthogonality_penalty(W=None, proj=None):
    """
    Returns the orthogonality penalty ||W^T W - I||.
    :param W: T.matrix, storing filters in column format
    """

    if proj is None:
        assert W
        proj = T.dot(W.T, W)
    
    I = T.identity_like(proj)
    penalty = proj - I

    return T.sum(T.sqrt(T.sum(penalty**2, axis=1)))
