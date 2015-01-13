import unittest
import numpy

import theano
import theano.tensor as T
import theano.sparse as S

from bilinear import sparse_masks
from ssrbm import ssrbm

batch_size = 5
nh = 20
bwh = 4
nv = 10*10*3
ns = nh * bwh

hval = numpy.random.rand(batch_size, nh)
sval = numpy.random.rand(batch_size, ns)
vval = numpy.random.rand(batch_size, nv)

Wh = sparse_masks.SparsityMask.unfactored_g(nh,nh,bwh,bwh).mask
Wv = numpy.random.rand(nv,ns)

hbias = numpy.random.rand(nh)
mu    = numpy.abs(numpy.random.rand(ns))
alpha = numpy.abs(numpy.random.rand(ns))
beta  = numpy.abs(numpy.random.rand(nv))

h = T.matrix('h')
s = T.matrix('s')
v = T.matrix('v')

xxx = T.matrix()
sigm = theano.function([xxx], T.nnet.sigmoid(xxx))

def quick_alloc():
    r = ssrbm.ssRBM(n_h=nh, n_v=nv, bw_h=bwh,
            iscales={'Wv': 0.1, 'hbias':0., 'alpha':1., 'mu':0., 'beta':1.},
            compile=False, batch_size=batch_size,
            sp_weight={'h':0.},
            sp_targ={'h':0.},
            sparse_hmask = sparse_masks.SparsityMask.unfactored_g(nh,nh,bwh,bwh),
            parametrize_sqrt_precision = False,
            l1={'Wh':0.,'Wv':0.},
            l2={'Wh':0.,'Wv':0.})
    r.Wv.set_value(Wv)
    r.hbias.set_value(hbias)
    r.mu.set_value(mu)
    r.alpha.set_value(alpha)
    r.beta.set_value(beta)
    return r

def test_ml_updates():
    r = ssrbm.ssRBM(n_h=nh, n_v=nv, bw_h=bwh,
            iscales={'Wv': 0.1, 'hbias':0., 'alpha':1., 'mu':0., 'beta':1.},
            compile=True, batch_size=batch_size,
            sp_weight={'h':0.},
            sp_targ={'h':0.},
            sparse_hmask = sparse_masks.SparsityMask.unfactored_g(nh,nh,bwh,bwh),
            parametrize_sqrt_precision = False,
            l1={'Wh':0.,'Wv':0.},
            l2={'Wh':0.,'Wv':0.})

    x = numpy.random.rand(batch_size, nv)
    r.batch_train_func(x)

def test_energy():
    r = quick_alloc()
    E = numpy.zeros(batch_size)
    # dummy numpy implementation
    for t in xrange(batch_size):

        for k in xrange(nv):
            E[t] += 0.5 * beta[k] * vval[t,k]**2

        for j in xrange(nh):
            for si in xrange(j*bwh, (j+1)*bwh):
                for k in xrange(nv):
                    E[t] -= vval[t,k] * Wv[k,si] * sval[t,si] * hval[t,j]
                E[t] += 0.5 * alpha[si] * sval[t,si]**2
                E[t] -= alpha[si] * mu[si] * sval[t,si] * hval[t,j]
                E[t] += 0.5 * alpha[si] * mu[si]**2 * hval[t,j]

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
            for si in xrange(j*bwh, (j+1)*bwh):
                fromv_si = 0
                for k in xrange(nv):
                    fromv_si += vval[t,k] * Wv[k,si]
                hmean[t,j] += 0.5 * 1./alpha[si] * fromv_si**2 
                hmean[t,j] += fromv_si * mu[si]
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
            for si in xrange(j*bwh, (j+1)*bwh):
                fromv_si = 0.
                for k in xrange(nv):
                    fromv_si += vval[t,k] * Wv[k,si]
                smean[t,si] = (1./alpha[si] * fromv_si + mu[si]) * hval[t,j]

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
        for j in xrange(nh):
            for si in xrange(j*bwh, (j+1)*bwh):
                temp = sval[t,si] * hval[t,j]
                for k in xrange(nv):
                    vmean[t,k] += 1./beta[k] * Wv[k,si] * temp

    # model implementation
    outsym = r.v_given_hs(h,s)
    theano_f   = theano.function([h,s], outsym)
    theano_val = theano_f(hval,sval)

    numpy.testing.assert_almost_equal(vmean, theano_val)
