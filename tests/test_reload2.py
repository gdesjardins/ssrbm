import numpy
import pickle
import md5
from ssrbm import pooled_ss_rbm
from pylearn2.utils import serial

def info(model, msg):
    print '**** %s ****' % msg
    print 'md5(Wv) = ', md5.new(model.Wv.get_value()).hexdigest()
    print 'rstate = ', model.theano_rng.rstate
    print 'state_updates:'
    for s in model.theano_rng.state_updates:
        print '\t', md5.new(s[0].get_value()).hexdigest()

load_r = pooled_ss_rbm.PooledSpikeSlabRBM(
        init_from = 'model_serial.pkl',
        lr={'start':1e-5, 'end':1e-5, 'type':'linear'},
        iscales={'Wv':0.01, 'vbias':0, 'hbias':0, 'alpha': 0., 'beta': 0}, 
        var_param_alpha = 'exp',
        var_param_beta = 'exp',
        sp_weight={'h':0},
        sp_targ={'h':0},
        flags={'use_cd':0})

rng = numpy.random.RandomState(1231)
x = rng.randint(0, 2, size=(load_r.batch_size, load_r.n_v))

info(load_r, '(reload) t=5')
for i in xrange(5): load_r.batch_train_func(x)
info(load_r, '(reload) t=10')


