import numpy
import pickle
import md5
from ssrbm import pooled_ss_rbm
from pylearn2.utils import serial

load_r = serial.load('model_serial.pkl')
print '**** reload (serial) ****'
print 'rstate = ', load_r.theano_rng.rstate
print 'state_updates:'
for s in load_r.theano_rng.state_updates:
    print '\t', md5.new(s[0].get_value()).hexdigest()
