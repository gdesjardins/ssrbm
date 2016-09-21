import numpy
from utils import sharedX, floatX, npy_floatX

def compute_ml_bias(data, eps=1e-5):
    """
    Computes the maximum likelihood solution for biases of RBM model with 0-weights. This
    should be used to initialize the visible biases.
    :param data: numpy.ndarray (training data).
    """
    datax = numpy.asarray(data, dtype=floatX)
    meanx  = numpy.mean(datax, axis=0)
    clipx = numpy.clip(meanx, eps, 1. - eps)
    return numpy.log(clipx / (1. - clipx))
