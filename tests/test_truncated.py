import numpy
import theano
from theano.sandbox import rng_mrg
from ssrbm.truncated import truncated_normal as tnorm
from utils import sharedX
import pylab as pl

rng = rng_mrg.MRG_RandomStreams(1231)

avg = sharedX(5., name='mean')
std = sharedX(1, name='std')
r = tnorm(size=(10000,), avg=avg, std=std,
        lbound=numpy.cast['float32'](-2),
        ubound=numpy.cast['float32'](-0.5),
        theano_rng=rng,
        dtype=theano.config.floatX)
f = theano.function([], r)
x = f()
import pdb; pdb.set_trace()
pl.hist(x)
pl.show()
