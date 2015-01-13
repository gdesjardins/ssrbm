# Copyright (c) 2013, Guillaume Desjardins.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   * Neither the name of the <organization> nor the
#     names of its contributors may be used to endorse or promote products
#     derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy
import cPickle

from scipy import sparse as sp
import copy
import theano
import pylearn
import pickle
import theano.tensor as T
import theano.sparse as S
from theano.tensor import nnet
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import function, shared
from jobman import make, make2
from pylearn2.models.model import Model
from pylearn.datasets.dataset import Dataset
from pylearn2.space import VectorSpace
from utils import tools
from utils import cost as utils_cost
from utils import sharedX, floatX, npy_floatX

from pylearn2.gui.patch_viewer import PatchViewer
from pylearn2.training_algorithms import default
from pylearn2.utils import serial

class ssRBM(Model):
    """Spike & Slab Restricted Boltzmann Machine (RBM)  """

    def load_params(self, model_path):
        fp = open(model_path, 'r')
        model = pickle.load(fp)
        self.Wv.set_value(model.Wv.get_value())
        self.Wh.set_value(model.Wh.get_value())
        self.hbias.set_value(model.hbias.get_value())
        self.mu.set_value(model.mu.get_value())
        self.alpha.set_value(model.alpha.get_value())
        self.beta.set_value(model.beta.get_value())
        # sync random number generators
        self.rng.set_state(model.rng.get_state())
        self.theano_rng.rstate = model.theano_rng.rstate
        for (self_rng_state, model_rng_state) in \
                zip(self.theano_rng.state_updates, 
                    model.theano_rng.state_updates):
            self_rng_state[0].set_value(model_rng_state[0].get_value())
        # reset timestamps
        self.batches_seen = model.batches_seen
        self.examples_seen = model.examples_seen
        fp.close()

    def __init__(self, 
            input=None, Wv=None, hbias=None,
            numpy_rng = None, theano_rng = None,
            n_h=100, n_v=100, bw_h=10, init_from=None,
            neg_sample_steps=1,
            lr = 1e-3, lr_anneal_coeff=0, lr_timestamp=None, lr_mults = {},
            iscales={}, clip_min={}, clip_max={}, l1 = {}, l2 = {},
            sp_moving_avg=0.98, sp_type='KL', sp_weight={}, sp_targ={},
            batch_size = 13,
            scalar_b = False,
            sparse_hmask = None, 
            learn_h_weights = False,
            unit_norm_filters = True,
            compile=True,
            parametrize_sqrt_precision=True,
            debug=False,
            seed=1241234,
            my_save_path=None):
        """
        :param n_h: number of h-hidden units
        :param n_v: number of visible units
        :param iscales: optional dictionary containing initialization scale for each parameter
        :param neg_sample_steps: number of sampling updates to perform in negative phase.
        :param l1: hyper-parameter controlling amount of L1 regularization
        :param l2: hyper-parameter controlling amount of L2 regularization
        :param batch_size: size of positive and negative phase minibatch
        :param compile: compile sampling and learning functions
        :param seed: seed used to initialize numpy and theano RNGs.
        """
        super(ssRBM,self).__init__()
        for k in ['mu','alpha','beta', 'Wv', 'hbias']: assert k in iscales.keys()
        for k in ['h']: assert k in sp_weight.keys()
        for k in ['h']: assert k in sp_targ.keys()

        ### make sure all parameters are floatX ###
        for (k,v) in l1.iteritems(): l1[k] = npy_floatX(v)
        for (k,v) in l2.iteritems(): l2[k] = npy_floatX(v)
        for (k,v) in sp_weight.iteritems(): sp_weight[k] = npy_floatX(v)
        for (k,v) in sp_targ.iteritems(): sp_targ[k] = npy_floatX(v)
        for (k,v) in clip_min.iteritems(): clip_min[k] = npy_floatX(v)
        for (k,v) in clip_max.iteritems(): clip_max[k] = npy_floatX(v)

        # dump initialization parameters to object
        for (k,v) in locals().iteritems():
            if k!='self': setattr(self,k,v)

        # allocate random number generators
        self.rng = numpy.random.RandomState(seed) if numpy_rng is None else numpy_rng
        self.theano_rng = RandomStreams(self.rng.randint(2**30)) if theano_rng is None else theano_rng

        ############### ALLOCATE PARAMETERS #################
        self.n_s = self.n_h * bw_h

        # allocate bilinear-weight matrices
        self.Wh = sharedX(sparse_hmask.mask, name='Wh')

        if Wv is None:
            wv_val =  self.rng.randn(n_v, self.n_s) * iscales['Wv']
            self.Wv = sharedX(wv_val, name='Wv')
        else:
            self.Wv = Wv

        # allocate shared variables for bias parameters
        if hbias is None:
            self.hbias = sharedX(iscales['hbias'] * numpy.ones(n_h), name='hbias') 
        else:
            self.hbias = hbias

        # mean (mu) and precision (alpha) parameters on s
        self.mu = sharedX(iscales['mu'] * numpy.ones(self.n_s), name='mu')
        self.alpha = sharedX(iscales['alpha'] * numpy.ones(self.n_s), name='alpha')
        self.alpha_prec = self.alpha**2 if parametrize_sqrt_precision else self.alpha

        # diagonal of precision matrix of visible units
        self.beta = sharedX(iscales['beta'] * numpy.ones(n_v), name='beta')
        self.beta_prec = self.beta**2 if parametrize_sqrt_precision else self.beta

        #### load layer 1 parameters from file ####
        if init_from:
            self.load_params(init_from)

        # allocate shared variable for persistent chain
        self.neg_v  = sharedX(self.rng.rand(batch_size, n_v), name='neg_v')
        self.neg_ev = sharedX(self.rng.rand(batch_size, n_v), name='neg_ev')
        self.neg_s  = sharedX(self.rng.rand(batch_size, self.n_s), name='neg_s')
        self.neg_h  = sharedX(self.rng.rand(batch_size, n_h), name='neg_h')
       
        # moving average values for sparsity
        self.sp_pos_v = sharedX(self.rng.rand(1,self.n_v), name='sp_pos_v')
        self.sp_pos_h = sharedX(self.rng.rand(1,self.n_h), name='sp_pog_h')

        # learning rate - implemented as shared parameter for GPU
        self.lr_shrd = sharedX(lr, name='lr_shrd')
        self.lr_mults_it = {}
        self.lr_mults_shrd = {}
        for (k,v) in lr_mults.iteritems():
            # make sure all learning rate multipliers are float64
            self.lr_mults_it[k] = tools.HyperParamIterator(lr_timestamp, lr_mults[k])
            self.lr_mults_shrd[k] = sharedX(self.lr_mults_it[k].value, 
                                            name='lr_mults_shrd'+k)

        # allocate symbolic variable for input
        self.input = T.matrix('input') if input is None else input
        
        # configure input-space (new pylearn2 feature?)
        self.input_space = VectorSpace(n_v)

        # counters used by pylearn2 trainers
        self.batches_seen = 0                    # incremented on every batch
        self.examples_seen = 0                   # incremented on every training example
        self.force_batch_size = batch_size  # force minibatch size

        self.error_record = []

        ## ESTABLISH LIST OF LEARNT MODEL PARAMETERS ##
        self.params = [self.Wv, self.hbias, self.mu, self.alpha, self.beta]
        if self.learn_h_weights:
            self.params += [self.Wh]
        
        if compile: self.do_theano()

    def do_theano(self):
        """ Compiles all theano functions needed to use the model"""

        init_names = dir(self)

        ###### All fields you don't want to get pickled (e.g., theano functions) should be created below this line

        # SAMPLING: NEGATIVE PHASE
        neg_updates = self.neg_sampling_updates(n_steps=self.neg_sample_steps)
        self.sample_neg_func = function([], [], updates=neg_updates, name='sample_neg_func')

        pos_updates = {}

        # determing maximum likelihood cost
        main_cost = [self.ml_cost(),
                     self.get_sparsity_cost(),
                     self.get_reg_cost(self.l2, self.l1)]
 
        ##
        # COMPUTE GRADIENTS WRT. TO ALL COSTS
        ##
        learning_grads = utils_cost.compute_gradients(*main_cost)

        ##
        # BUILD UPDATES DICTIONARY
        ##
        learning_updates = utils_cost.get_updates(
                learning_grads,
                self.lr_shrd,
                multipliers = self.lr_mults_shrd)
        if self.learn_h_weights:
            learning_updates[self.Wh] *= self.sparse_hmask.mask
        learning_updates.update(pos_updates)
      
        # build theano function to train on a single minibatch
        self.batch_train_func = function([self.input], [],
                                         updates=learning_updates, name='train_rbm_func')

        # enforce constraints function
        constraint_updates = {}
        ## clip parameters to maximum values (if applicable)
        for (k,v) in self.clip_max.iteritems():
            assert k in [param.name for param in self.params]
            param = getattr(self, k)
            constraint_updates[param] = T.clip(param, param, v)
        ## clip parameters to minimum values (if applicable)
        for (k,v) in self.clip_min.iteritems():
            assert k in [param.name for param in self.params]
            param = getattr(self, k)
            constraint_updates[param] = T.clip(constraint_updates.get(param, param), v, param)
        ## Residual variance on beta is scalar valued
        if self.scalar_b:
            beta = constraint_updates.get(self.beta, self.beta)
            constraint_updates[self.beta] = T.mean(beta) * T.ones_like(beta)
        # constraint filters to have unit norm
        if self.unit_norm_filters:
            Wv = constraint_updates.get(self.Wv, self.Wv)
            constraint_updates[self.Wv] = Wv / T.sqrt(T.sum(Wv**2, axis=0))
        self.enforce_constraints = theano.function([],[], updates=constraint_updates)

        ###### All fields you don't want to get pickled should be created above this line
        final_names = dir(self)
        self.register_names_to_del( [ name for name in (final_names) if name not in init_names ])

        # Before we start learning, make sure constraints are enforced
        self.enforce_constraints()

    def learn(self, dataset, batch_size):

        x = dataset.get_batch_design(batch_size, include_labels=False)
        self.learn_mini_batch(x)

        # accounting...
        self.examples_seen += self.batch_size
        self.batches_seen += 1

        # modify learning rate multipliers
        for (k, iter) in self.lr_mults_it.iteritems():
            if iter.next():
                print 'self.batches_seen = ', self.batches_seen
                self.lr_mults_shrd[k].set_value(iter.value)
                print 'lr_mults_shrd[%s] = %f' % (k,iter.value)

        self.enforce_constraints()

        # save to different path each epoch
        if self.my_save_path and self.batches_seen%1000==0:
            fname = self.my_save_path + '_e%i.pkl' % (self.batches_seen/1000)
            print 'Saving to %s ...' %fname,
            serial.save(fname, self)
            print 'done'


    def learn_mini_batch(self, x):

        # anneal learning rate
        self.lr_shrd.set_value(self.lr / (1. + self.lr_anneal_coeff * self.batches_seen))

        # perform negative phase sampling
        self.sample_neg_func()
        if self.debug and (
            numpy.isnan(self.neg_h.get_value()).any() or
            numpy.isnan(self.neg_s.get_value()).any() or
            numpy.isnan(self.neg_v.get_value()).any()):
            import pdb; pdb.set_trace()

        # update parameters
        self.batch_train_func(x)


    def energy(self, h_sample, s_sample, v_sample):
        """
        Computes energy for a given configuration of (g,h,v,x,y).
        :param h_sample: T.matrix of shape (batch_size, n_h)
        :param s_sample: T.matrix of shape (batch_size, bw_h * n_h)
        :param v_sample: T.matrix of shape (batch_size, n_v)
        """
        energy = -T.sum(s_sample * 
                        T.dot(v_sample, self.Wv) *
                        T.dot(h_sample, self.Wh), axis=1)
        energy += T.sum(0.5 * self.alpha_prec * s_sample**2, axis=1)
        energy += T.sum(0.5 * self.beta_prec * v_sample**2, axis=1)

        energy -= T.sum(self.alpha_prec * self.mu * s_sample *
                        T.dot(h_sample, self.Wh), axis=1)

        energy += T.sum(0.5 * self.alpha_prec * self.mu**2 *
                        T.dot(h_sample, self.Wh), axis=1)

        energy -= T.dot(h_sample, self.hbias)

        return energy

    def __call__(self, v, output_type='hs'):
        assert output_type in ['h', 'hs']
        h_mean = self.h_given_v(v)
        s_mean = self.s_given_hv(h_mean, v_sample)
        output_prods = {
                'h': h_mean,
                'hs': T.dot(h_mean, self.Wh) * s_mean
                }
        return output_prods[output_type]

    ######################################
    # MATH FOR CONDITIONAL DISTRIBUTIONS #
    ######################################

    def h_given_v(self, v_sample):
        """
        Compute mean activation of h given v.
        :param v_sample: T.matrix of shape (batch_size, n_v matrix)
        """
        from_v = T.dot(v_sample, self.Wv)

        temp =  0.5 * 1./self.alpha_prec * from_v**2
        temp += from_v * self.mu
        h_mean = T.dot(temp, self.Wh.T) + self.hbias
        
        return T.nnet.sigmoid(h_mean)

    def sample_h_given_v(self, v_sample):
        """
        Generates sample from p(h|v)
        """
        h_mean = self.h_given_v(v_sample)
        h_sample = self.theano_rng.binomial(size=(self.batch_size,self.n_h),
                                            n=1, p=h_mean, dtype=floatX)
        return h_sample

    def s_given_hv(self, h_sample, v_sample):
        from_h = T.dot(h_sample, self.Wh)
        from_v = T.dot(v_sample, self.Wv)
        s_mean = (1./self.alpha_prec * from_v + self.mu) * from_h
        return s_mean

    def sample_s_given_hv(self, h_sample, v_sample):
        s_mean = self.s_given_hv(h_sample, v_sample)
        s_sample = self.theano_rng.normal(
                size=(self.batch_size, self.n_s),
                avg = s_mean, 
                std = T.sqrt(1./self.alpha_prec), dtype=floatX)
        return s_sample

    def v_given_hs(self, h_sample, s_sample):
        """
        Computes the mean-activation of visible units, given all other variables.
        :param h_sample: T.matrix of shape (batch_size, n_h)
        :param s_sample: T.matrix of shape (batch_size, n_s)
        """
        from_h = T.dot(h_sample, self.Wh)
        v_mean = 1./self.beta_prec * T.dot(s_sample * from_h, self.Wv.T)
        return v_mean

    def sample_v_given_hs(self, h_sample, s_sample):
        v_mean = self.v_given_hs(h_sample, s_sample)
        v_sample = self.theano_rng.normal(
                size=(self.batch_size, self.n_v),
                avg = v_mean, 
                std = T.sqrt(1./self.beta_prec), dtype=floatX)
        return v_sample


    ##################
    # SAMPLING STUFF #
    ##################

    def neg_sampling(self, h_sample, s_sample, v_sample, n_steps=1):
        """
        Gibbs step for negative phase, which alternates: 
        p(h|b,g,v), p(s|b,g,h,v) and p(v|b,g,h,s)
        :param f_sample: T.matrix of shape (batch_size, n_f)
        :param g_sample: T.matrix of shape (batch_size, n_g)
        :param h_sample: T.matrix of shape (batch_size, n_h)
        :param s_sample: T.matrix of shape (batch_size, n_s)
        :param v_sample: T.matrix of shape (batch_size, n_v)
        :param n_steps: number of Gibbs updates to perform in negative phase.
        """

        def gibbs_iteration(h1, s1, v1):
            h2 = self.sample_h_given_v(v1)
            s2 = self.sample_s_given_hv(h2, v1)
            v2 = self.sample_v_given_hs(h2, s2)
            return [h2, s2, v2]

        [new_h, new_s, new_v] , updates = theano.scan(
                gibbs_iteration,
                outputs_info = [h_sample, s_sample, v_sample],
                n_steps=n_steps)

        return [new_h[-1], new_s[-1], new_v[-1]]

    def neg_sampling_updates(self, n_steps=1):
        """
        Implements the negative phase, generating samples from p(h,s,v).
        :param n_steps: scalar, number of Gibbs steps to perform.
        """
        [new_h, new_s, new_v] =  self.neg_sampling(self.neg_h, self.neg_s,
                                  self.neg_v, n_steps=n_steps)

        # we want to plot the expected value of the samples
        new_ev = self.v_given_hs(new_h, new_s)

        updates = {self.neg_h : new_h,
                   self.neg_s : new_s,
                   self.neg_v : new_v,
                   self.neg_ev: new_ev}

        return updates

    def ml_cost(self):
        """
        Variational approximation to the maximum likelihood positive phase.
        :param v: T.matrix of shape (batch_size, n_v), training examples
        :return: tuple (cost, gradient)
        """
        pos_h = self.h_given_v(self.input)
        pos_s = self.s_given_hv(pos_h, self.input)

        pos_cost = T.sum(self.energy(pos_h, pos_s, self.input)) 
        neg_cost = T.sum(self.energy(self.neg_h, self.neg_s, self.neg_v))
        batch_cost = pos_cost - neg_cost
        cost = batch_cost / self.batch_size

        # build gradient of cost with respect to model parameters
        cte = [pos_h, pos_s, self.neg_h, self.neg_s, self.neg_v]

        return utils_cost.Cost(cost, self.params, cte)

    def get_sparsity_cost(self):

        # update mean activation using exponential moving average
        hack_h   = self.h_given_v(self.sp_pos_v)

        # define loss based on value of sp_type
        if self.sp_type == 'KL':
            eps = 1./self.batch_size
            loss = lambda targ, val: - targ * T.log(eps + val) - (1.-targ) * T.log(1. - val + eps)
        elif self.sp_type.startswith('Lee07'):
            loss = lambda targ, val: abs(targ - val)
        else:
            raise NotImplementedError('Sparsity type %s is not implemented' % self.sp_type)

        cost = T.zeros((), dtype=floatX)

        params = []
        if self.sp_weight['h']: 
            cost += self.sp_weight['h']  * T.sum(loss(self.sp_targ['h'], hack_h.mean(axis=0)))
            params += [self.hbias]

        if self.sp_type in ['KL','Lee07'] and self.sp_weight['h']:
            params += [self.Wv, self.alpha, self.mu]

        return utils_cost.Cost(cost, params)


    ##############################
    # GENERIC OPTIMIZATION STUFF #
    ##############################
    def get_reg_cost(self, l2=None, l1=None):
        """
        Builds the symbolic expression corresponding to first-order gradient descent
        of the cost function ``cost'', with some amount of regularization defined by the other
        parameters.
        :param l2: dict containing amount of L2 regularization for Wg, Wh and Wv
        :param l1: dict containing amount of L1 regularization for Wg, Wh and Wv
        """
        cost = T.zeros((), dtype=floatX)
        params = []

        for p in self.params:

            if l1.get(p.name, 0):
                cost += l1[p.name] * T.sum(abs(p))
                params += [p]

            if l2.get(p.name, 0):
                cost += l2[p.name] * T.sum(p**2)
                params += [p]
            
        return utils_cost.Cost(cost, params)

    def monitor_matrix(self, w, name=None):
        if name is None: assert hasattr(w, 'name')
        name = name if name else w.name

        return {name + '.min':  w.min(axis=[0,1]),
                name + '.max':  w.max(axis=[0,1]),
                name + '.absmean': abs(w).mean(axis=[0,1])}

    def monitor_vector(self, b, name=None):
        if name is None: assert hasattr(b, 'name')
        name = name if name else b.name

        return {name + '.min':  b.min(),
                name + '.max':  b.max(),
                name + '.absmean': abs(b).mean()}

    def get_monitoring_channels(self, x):
        chans = {}
        chans.update(self.monitor_matrix(self.Wv))
        chans.update(self.monitor_vector(self.hbias))
        chans.update(self.monitor_vector(self.alpha))
        chans.update(self.monitor_vector(self.mu))
        chans.update(self.monitor_vector(self.beta))
        chans.update(self.monitor_matrix(self.neg_h))
        chans.update(self.monitor_matrix(self.neg_s))
        chans.update(self.monitor_matrix(self.neg_v))
        return chans

class TrainingAlgorithm(default.DefaultTrainingAlgorithm):

    def setup(self, model, dataset):
        super(TrainingAlgorithm, self).setup(model, dataset)
