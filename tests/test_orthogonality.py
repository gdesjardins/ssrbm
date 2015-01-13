import numpy
import theano
import theano.tensor as T
from theano.sandbox.cuda.basic_ops import gpu_from_host
from ssrbm import orthogonality
import copy

def test_orthogonal_pool_cost():
    (n_h, bw_s, n_v) = (1,3,3)
    Wval = numpy.random.rand(n_v, n_h * bw_s).astype('float32')
    W = theano.shared(Wval, name='W')

    # theano implementation
    penalty = orthogonality.orthogonal_pools(W, bw_s)
    f = theano.function([], [penalty])

    # reference implementation
    temp = (numpy.dot(Wval.T, Wval) - numpy.identity(n_h * bw_s))**2
    ref = 0.
    for i in xrange(0, n_h, bw_s):
        ref += temp[i:i+bw_s, i:i+bw_s].sum()

    numpy.testing.assert_almost_equal(ref, f(), decimal=4)
