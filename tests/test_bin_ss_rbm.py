import unittest
import numpy

import theano
import theano.tensor as T
import theano.sparse as S

from bilinear import sparse_masks
from ssrbm import bin_ss_rbm

batch_size = 5
nh = 13
ns = nh
nv = 21

hval = numpy.random.rand(batch_size, nh)
sval = numpy.random.rand(batch_size, ns)
vval = numpy.random.rand(batch_size, nv)
Wv = numpy.random.rand(nv,ns)

hbias = numpy.random.rand(nh)
mu    = numpy.abs(numpy.random.rand(ns))
alpha = numpy.abs(numpy.random.rand(ns))
vbias = numpy.random.rand(nv)

h = T.matrix('h')
s = T.matrix('s')
v = T.matrix('v')

xxx = T.matrix()
sigm = theano.function([xxx], T.nnet.sigmoid(xxx))

def quick_alloc(compile=True):
    r = bin_ss_rbm.BinarySpikeSlabRBM(n_h=nh, n_v=nv,
            lr = {'type': 'linear', 'start': 0.1, 'end':0.1},
            iscales={'Wv': 0.1, 'vbias':0., 'hbias':0., 'alpha':1., 'mu':0.},
            compile=compile,
            batch_size=batch_size,
            sp_weight={'h':0.},
            sp_targ={'h':0.},
            parametrize_sqrt_precision = False,
            l1={'Wh':0.,'Wv':0.},
            l2={'Wh':0.,'Wv':0.})
    r.Wv.set_value(Wv)
    r.hbias.set_value(hbias)
    r.vbias.set_value(vbias)
    r.mu.set_value(mu)
    r.alpha.set_value(alpha)
    return r

def test_ml_updates():
    r = quick_alloc(compile=True)
    x = numpy.random.rand(batch_size, nv)
    r.batch_train_func(x)

def test_energy():
    r = quick_alloc()
    E = numpy.zeros(batch_size)
    # dummy numpy implementation
    for t in xrange(batch_size):

        for j in xrange(nh):
            for k in xrange(nv):
                E[t] -= vval[t,k] * Wv[k,j] * sval[t,j] * hval[t,j]
            E[t] += 0.5 * alpha[j] * sval[t,j]**2
            E[t] -= alpha[j] * mu[j] * sval[t,j] * hval[t,j]
            E[t] += 0.5 * alpha[j] * mu[j]**2 * hval[t,j]

        for k in xrange(nv):
            E[t] -=  vbias[k] * vval[t,k]
        for j in xrange(nh):
            E[t] -= hbias[j] * hval[t,j]

    # model implementation
    outsym = r.energy(h,s,v)
    fe = theano.function([h,s,v], outsym)
    theano_E = fe(hval,sval,vval)

    numpy.testing.assert_almost_equal(E, theano_E)

def test_h_given_v():
    r = quick_alloc()
    hmean = numpy.zeros((batch_size, nh))

    # dummy numpy implementation
    for t in xrange(batch_size):
        for j in xrange(nh):
            fromv = 0
            for k in xrange(nv):
                fromv += vval[t,k] * Wv[k,j]
            hmean[t,j] += 0.5 * 1./alpha[j] * fromv**2 
            hmean[t,j] += fromv * mu[j]
            hmean[t,j] += hbias[j]

    # model implementation
    outsym = r.h_given_v(v)
    theano_f   = theano.function([v], outsym)
    theano_val = theano_f(vval)

    numpy.testing.assert_almost_equal(sigm(hmean), theano_val)
 
def test_s_given_hv():
    r = quick_alloc()
    smean = numpy.zeros((batch_size, ns))

    # dummy numpy implementation
    for t in xrange(batch_size):
        for j in xrange(nh):
            fromv = 0.
            for k in xrange(nv):
                fromv += vval[t,k] * Wv[k,j]
            smean[t,j] = (1./alpha[j] * fromv + mu[j]) * hval[t,j]

    # model implementation
    outsym = r.s_given_hv(h,v)
    theano_f   = theano.function([h,v], outsym)
    theano_val = theano_f(hval,vval)

    numpy.testing.assert_almost_equal(smean, theano_val)

def test_v_given_hs():
    r = quick_alloc()
    vmean = numpy.zeros((batch_size, nv))

    # dummy numpy implementation
    for t in xrange(batch_size):
        for k in xrange(nv):
            for j in xrange(nh):
                vmean[t,k] +=  Wv[k,j] * sval[t,j] * hval[t,j]
            vmean[t,k] += vbias[k]

    # model implementation
    outsym = r.v_given_hs(h,s)
    theano_f   = theano.function([h,s], outsym)
    theano_val = theano_f(hval,sval)

    numpy.testing.assert_almost_equal(sigm(vmean), theano_val)
