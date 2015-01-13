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
import pickle

import theano
import theano.tensor as T
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import function, shared

from pylearn2.training_algorithms import default
from pylearn2.utils import serial
from pylearn2.base import Block
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace

from utils import tools
from utils import rbm_utils
from utils import cost as utils_cost
from utils import sharedX, floatX, npy_floatX

class S4RBM(Model, Block):
    """Spike & Slab Restricted Boltzmann Machine (RBM)  """

    def load_params(self, model_path):
        fp = open(model_path, 'r')
        model = pickle.load(fp)
        for (p, mp) in zip(self.params(), model.params()):
            p.set_value(mp.get_value())
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
        self.iter.set_value(model.iter.get_value())
        fp.close()

    def __init__(self, 
            numpy_rng = None, theano_rng = None,
            n_h=100, n_v=100,
            init_from=None,
            layer1_model=None,
            layer2_model=None,
            neg_sample_steps=1,
            lr=None, lr_timestamp=None, lr_mults = {},
            iscales={}, clip_min={}, clip_max={}, l1 = {}, l2 = {},
            sp_type='kl', sp_weight={}, sp_targ={},
            batch_size = 13,
            compile=True,
            debug=False,
            seed=1241234,
            my_save_path=None, save_at=None, save_every=None,
            flags = {},
            max_updates = 5e5):
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
        Model.__init__(self)
        Block.__init__(self)
        assert lr is not None
        assert layer1_model
        for k in ['Wv', 'vbias', 'hbias']: assert k in iscales.keys()
        iscales.setdefault('mu', 1.)
        iscales.setdefault('alpha', 0.)
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
        wv_val =  self.rng.randn(n_v, self.n_h) * iscales['Wv']
        self.Wv = sharedX(wv_val, name='Wv')
        # allocate shared variables for bias parameters
        self.vbias = sharedX(iscales['vbias'] * numpy.ones(n_v), name='vbias') 
        # allocate shared variables for bias parameters
        self.hbias = sharedX(iscales['hbias'] * numpy.ones(n_h), name='hbias') 

        # mean (mu) and precision (alpha) parameters on s
        self.mu_h = sharedX(iscales['mu_h'] * numpy.ones(self.n_h), name='mu_h')
        self.mu_v = sharedX(iscales['mu_v'] * numpy.ones(self.n_v), name='mu_v')
        self.alpha_h = sharedX(iscales['alpha_h'] * numpy.ones(self.n_h), name='alpha_h')
        self.alpha_v = sharedX(iscales['alpha_v'] * numpy.ones(self.n_v), name='alpha_v')
        self.alpha_h_prec = T.exp(self.alpha_h)
        self.alpha_v_prec = T.exp(self.alpha_v)

        #### load layer 1 parameters from file ####
        if init_from:
            self.load_params(init_from)

        # allocate shared variable for persistent chain
        self.neg_v  = sharedX(self.rng.rand(batch_size, n_v), name='neg_v')
        self.neg_ev = sharedX(self.rng.rand(batch_size, n_v), name='neg_ev')
        self.neg_s  = sharedX(self.rng.rand(batch_size, self.n_h), name='neg_s')
        self.neg_x  = sharedX(self.rng.rand(batch_size, self.n_v), name='neg_x')
        self.neg_h  = sharedX(self.rng.rand(batch_size, n_h), name='neg_h')
       
        # moving average values for sparsity
        self.sp_pos_v = sharedX(self.rng.rand(1,self.n_v), name='sp_pos_v')
        self.sp_pos_x = sharedX(self.rng.rand(1,self.n_v), name='sp_pos_x')
        self.sp_pos_h = sharedX(self.rng.rand(1,self.n_h), name='sp_pog_h')

        # learning rate, with deferred 1./t annealing
        self.iter = sharedX(0.0, name='iter')

        if lr['type'] == 'anneal':
            num = lr['init'] * lr['start'] 
            denum = T.maximum(lr['start'], lr['slope'] * self.iter)
            self.lr = T.maximum(lr['floor'], num/denum) 
        elif lr['type'] == 'linear':
            lr_start = npy_floatX(lr['start'])
            lr_end   = npy_floatX(lr['end'])
            self.lr = lr_start + self.iter * (lr_end - lr_start) / npy_floatX(self.max_updates)
        else:
            raise ValueError('Incorrect value for lr[type]')

        # learning rate - implemented as shared parameter for GPU
        self.lr_mults_it = {}
        self.lr_mults_shrd = {}
        for (k,v) in lr_mults.iteritems():
            # make sure all learning rate multipliers are float64
            self.lr_mults_it[k] = tools.HyperParamIterator(lr_timestamp, lr_mults[k])
            self.lr_mults_shrd[k] = sharedX(self.lr_mults_it[k].value, 
                                            name='lr_mults_shrd'+k)

        # allocate symbolic variable for input
        self.input_v = T.matrix('input_v')
        self.input_x = T.matrix('input_x')
        
        # configure input-space (new pylearn2 feature?)
        self.input_space = VectorSpace(n_v)
        self.output_space = VectorSpace(n_h)

        self.batches_seen = 0                    # incremented on every batch
        self.examples_seen = 0                   # incremented on every training example
        self.force_batch_size = batch_size  # force minibatch size

        self.error_record = []
        
        if compile: self.do_theano()

    def params(self):
        """
        Returns a list of learnt model parameters.
        """
        return [self.Wv, self.vbias, self.hbias,
                self.alpha_h, self.mu_h,
                self.alpha_v, self.mu_v]

    def init_preprocessor(self):
        # load layer 1 from pickle file
        fp = open(self.layer1_model, 'r')
        self.l1model = pickle.load(fp)
        fp.close()

        # preprocessor for input data
        x = T.matrix()
        h1 = self.l1model.sample_h_given_v(x)
        s1 = self.l1model.sample_s_given_hv(h1,x)

        ###
        # LAYER 2
        ###
        if hasattr(self, 'layer2_model'):
            fp = open(self.layer2_model, 'r')
            self.l2model = pickle.load(fp)
            fp.close()

            # preprocessor for input data
            h2 = self.l2model.sample_h_given_vx(h1, s1)
            s2 = self.l2model.sample_s_given_vxh(h1, s1, h2)
            self.preproc = theano.function([x], [h2,s2])
 
            # initialize visible (mu,alpha) parameters from layer 1 latent variables
            self.mu_v.set_value(self.l2model.mu_h.get_value())
            self.alpha_v.set_value(self.l2model.alpha_h.get_value())
            self.vbias.set_value(self.l2model.hbias.get_value())
        else:
            # initialize visible (mu,alpha) parameters from layer 1 latent variables
            self.mu_v.set_value(self.l1model.mu.get_value())
            self.alpha_v.set_value(self.l1model.alpha.get_value())
            self.vbias.set_value(self.l1model.hbias.get_value())
            self.preproc = theano.function([x], [h1,s1])


    def do_theano(self):
        """ Compiles all theano functions needed to use the model"""

        init_names = dir(self)

        self.init_preprocessor()

        ###### All fields you don't want to get pickled (e.g., theano functions) should be created below this line
        # SAMPLING: NEGATIVE PHASE
        neg_updates = self.neg_sampling_updates(
                n_steps=self.neg_sample_steps,
                use_pcd=not self.flags['use_cd'])

        # determing maximum likelihood cost
        ml_cost = self.ml_cost(pos_v = self.input_v,
                               pos_x = self.input_x,
                               neg_v = neg_updates[self.neg_v],
                               neg_x = neg_updates[self.neg_x])
        main_cost = [ml_cost,
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
                self.lr,
                multipliers = self.lr_mults_shrd)
        learning_updates.update(neg_updates)
        learning_updates.update({self.iter: self.iter+1})
      
        # build theano function to train on a single minibatch
        self.batch_train_func = function([self.input_v, self.input_x], [],
                                         updates=learning_updates, name='train_rbm_func')

        # enforce constraints function
        constraint_updates = {}
        ## clip parameters to maximum values (if applicable)
        for (k,v) in self.clip_max.iteritems():
            assert k in [param.name for param in self.params()]
            param = getattr(self, k)
            constraint_updates[param] = T.clip(param, param, v)
        ## clip parameters to minimum values (if applicable)
        for (k,v) in self.clip_min.iteritems():
            assert k in [param.name for param in self.params()]
            param = getattr(self, k)
            constraint_updates[param] = T.clip(constraint_updates.get(param, param), v, param)
        # constraint filters to have unit norm
        if self.flags.get('weight_norm', None):
            wv = constraint_updates.get(self.Wv, self.Wv)
            wv_norm = T.sqrt(T.sum(wv**2, axis=0))
            if self.flags['weight_norm'] == 'unit':
                constraint_updates[self.Wv] = wv / wv_norm
            elif self.flags['weight_norm'] == 'max_unit':
                constraint_updates[self.Wv] = wv / wv_norm * T.minimum(wv_norm, 1.0)
        self.enforce_constraints = theano.function([],[], updates=constraint_updates)

        ###### All fields you don't want to get pickled should be created above this line
        final_names = dir(self)
        self.register_names_to_del( [ name for name in (final_names) if name not in init_names ])

        # Before we start learning, make sure constraints are enforced
        self.enforce_constraints()

    def train_batch(self, dataset, batch_size):

        x = dataset.get_batch_design(batch_size, include_labels=False)
        [x_spike, x_slab] = self.preproc(x)
        self.batch_train_func(x_spike, x_slab)

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
        if self.my_save_path and \
           (self.batches_seen in self.save_at or
            self.batches_seen % self.save_every == 0):
            fname = self.my_save_path + '_e%i.pkl' % self.batches_seen
            print 'Saving to %s ...' % fname,
            serial.save(fname, self)
            print 'done'

        return self.batches_seen < self.max_updates

    def energy(self, v_sample, x_sample, h_sample, s_sample):
        """
        Computes energy for a given configuration of (g,h,v,x,y).
        :param v_sample: T.matrix of shape (batch_size, n_v)
        :param x_sample: T.matrix of shape (batch_size, n_v)
        :param h_sample: T.matrix of shape (batch_size, n_h)
        :param s_sample: T.matrix of shape (batch_size, n_h)
        """
        from_v = T.dot(x_sample * v_sample, self.Wv)
        energy = -T.sum(from_v * s_sample * h_sample, axis=1)
        energy += T.sum(0.5 * self.alpha_h_prec * s_sample**2, axis=1)
        energy += T.sum(0.5 * self.alpha_v_prec * x_sample**2, axis=1)
        energy -= T.sum(self.alpha_h_prec * self.mu_h * s_sample * h_sample, axis=1)
        energy -= T.sum(self.alpha_v_prec * self.mu_v * x_sample * v_sample, axis=1)
        energy += T.sum(0.5 * self.alpha_h_prec * self.mu_h**2 * h_sample, axis=1)
        energy += T.sum(0.5 * self.alpha_v_prec * self.mu_v**2 * v_sample, axis=1)
        energy -= T.dot(h_sample, self.hbias)
        energy -= T.dot(v_sample, self.vbias)
        return energy

    def free_energy(self, v_sample, x_sample):
        fe = 0.
        fe -= T.dot(v_sample, self.vbias)
        fe += T.sum(0.5 * self.alpha_v_prec * x_sample**2, axis=1)
        fe -= T.sum(self.alpha_v_prec * self.mu_v * x_sample * v_sample, axis=1)
        fe += T.sum(0.5 * self.alpha_v_prec * self.mu_v**2 * v_sample, axis=1)
        fe -= T.sum(0.5 * T.log(2*numpy.pi / self.alpha_h_prec))
        h_mean = self.h_given_vx_input(v_sample, x_sample)
        fe -= T.sum(T.nnet.softplus(h_mean), axis=1)
        return fe

    def __call__(self, v, output_type='h'):

        if hasattr(self, 'layer1_model'):
            h1 = self.l1model.h_given_v(v)
            s1 = self.l1model.s_given_hv(h1, v)

        if hasattr(self, 'layer2_model'):
           # preprocessor for input data
            h2 = self.l2model.h_given_vx(h1, s1)
            s2 = self.l2model.s_given_vxh(h1, s1, h2)
            h3 = self.h_given_vx(h2, s2)
            return T.horizontal_stack(h1,h2,h3)
        else:            
            h2 = self.h_given_vx(h1, s1)
            return T.horizontal_stack(h1,h2)

    ######################################
    # MATH FOR CONDITIONAL DISTRIBUTIONS #
    ######################################

    ###
    # HIDDENS GIVEN VISIBLE
    ###
    def h_given_vx_input(self, v_sample, x_sample):
        """
        Compute mean activation of h given v, x.
        :param v_sample: T.matrix of shape (batch_size, n_v matrix)
        :param x_sample: T.matrix of shape (batch_size, n_v matrix)
        """
        from_v = T.dot(x_sample * v_sample, self.Wv)
        h_mean  = 0.5 * 1./self.alpha_h_prec * from_v**2
        h_mean += from_v * self.mu_h
        h_mean += self.hbias
        return h_mean
    
    def h_given_vx(self, v_sample, x_sample):
        h_mean = self.h_given_vx_input(v_sample, x_sample)
        return T.nnet.sigmoid(h_mean)

    def sample_h_given_vx(self, v_sample, x_sample):
        """
        Generates sample from p(h|v,x)
        """
        h_mean = self.h_given_vx(v_sample, x_sample)
        h_sample = self.theano_rng.binomial(size=(v_sample.shape[0],self.n_h),
                                            n=1, p=h_mean, dtype=floatX)
        return h_sample

    def s_given_vxh(self, v_sample, x_sample, h_sample):
        from_v = T.dot(x_sample * v_sample, self.Wv)
        s_mean = (1./self.alpha_h_prec * from_v + self.mu_h) * h_sample
        return s_mean

    def sample_s_given_vxh(self, v_sample, x_sample, h_sample):
        s_mean = self.s_given_vxh(v_sample, x_sample, h_sample)
        s_sample = self.theano_rng.normal(
                size=(v_sample.shape[0], self.n_h),
                avg = s_mean, 
                std = T.sqrt(1./self.alpha_h_prec), dtype=floatX)
        return s_sample

    ###
    # VISIBLE GIVEN HIDDENS
    ###
    def v_given_hs_input(self, h_sample, s_sample):
        """
        Compute mean activation of v given (h,s).
        :param h_sample: T.matrix of shape (batch_size, n_h matrix)
        :param s_sample: T.matrix of shape (batch_size, n_h matrix)
        """
        from_h = T.dot(s_sample * h_sample, self.Wv.T)
        v_mean  = 0.5 * 1./self.alpha_v_prec * from_h**2
        v_mean += from_h * self.mu_v
        v_mean += self.vbias
        return v_mean
    
    def v_given_hs(self, h_sample, s_sample):
        v_mean = self.v_given_hs_input(h_sample, s_sample)
        return T.nnet.sigmoid(v_mean)

    def sample_v_given_hs(self, h_sample, s_sample):
        """
        Generates sample from p(v|h,s)
        """
        v_mean = self.v_given_hs(h_sample, s_sample)
        v_sample = self.theano_rng.binomial(size=(h_sample.shape[0], self.n_v),
                                            n=1, p=v_mean, dtype=floatX)
        return v_sample

    def x_given_vhs(self, v_sample, h_sample, s_sample):
        from_h = T.dot(s_sample * h_sample, self.Wv.T)
        x_mean = (1./self.alpha_v_prec * from_h + self.mu_v) * v_sample
        return x_mean

    def sample_x_given_vhs(self, v_sample, h_sample, s_sample):
        x_mean = self.x_given_vhs(v_sample, h_sample, s_sample)
        x_sample = self.theano_rng.normal(
                size=(v_sample.shape[0], self.n_v),
                avg = x_mean, 
                std = T.sqrt(1./self.alpha_v_prec), dtype=floatX)
        return x_sample


    ##################
    # SAMPLING STUFF #
    ##################

    def neg_sampling(self, v_sample, x_sample, n_steps=1):
        """
        p(h|v,x), p(s|v,x,h), p(v|h,s) and p(x|v,h,s)
        :param v_sample: T.matrix of shape (batch_size, n_v)
        :param n_steps: number of Gibbs updates to perform in negative phase.
        """
        def gibbs_iteration(v1,x1):
            h2 = self.sample_h_given_vx(v1, x1)
            s2 = self.sample_s_given_vxh(v1, x1, h2)
            v2 = self.sample_v_given_hs(h2, s2)
            x2 = self.sample_x_given_vhs(v2, h2, s2)
            return [v2, x2, h2, s2]

        [new_v, new_x, new_h, new_s] , updates = theano.scan(
                gibbs_iteration,
                outputs_info = [v_sample, x_sample, None, None],
                n_steps=n_steps)

        return [new_v[-1], new_x[-1], new_h[-1], new_s[-1]]

    def neg_sampling_updates(self, n_steps=1, use_pcd=True):
        """
        Implements the negative phase, generating samples from p(h,s,v).
        :param n_steps: scalar, number of Gibbs steps to perform.
        """
        init_v = self.neg_v if use_pcd else self.input_v
        init_x = self.neg_x if use_pcd else self.input_x
        [new_v, new_x, new_h, new_s] = self.neg_sampling(init_v, init_x, n_steps=n_steps)

        # we want to plot the expected value of the samples
        new_ev = self.v_given_hs(new_h, new_s)

        updates = {self.neg_v : new_v,
                   self.neg_x : new_x,
                   self.neg_h : new_h,
                   self.neg_s : new_s,
                   self.neg_ev: new_ev}

        return updates

    def ml_cost(self, pos_v, pos_x, neg_v, neg_x):
        pos_cost = T.sum(self.free_energy(pos_v, pos_x))
        neg_cost = T.sum(self.free_energy(neg_v, neg_x))
        batch_cost = pos_cost - neg_cost
        cost = batch_cost / self.batch_size
        return utils_cost.Cost(cost, self.params(),
                               [pos_v, pos_x, neg_v, neg_x])

    def get_sparsity_cost(self):

        # update mean activation using exponential moving average
        hack_h = self.h_given_vx(self.sp_pos_v, self.sp_pos_x)

        # define loss based on value of sp_type
        if self.sp_type == 'kl':
            eps = npy_floatX(1./self.batch_size)
            loss = lambda targ, val: - npy_floatX(targ) * T.log(eps + val) \
                                     - npy_floatX(1-targ) * T.log(1 - val + eps)
        else:
            raise NotImplementedError('Sparsity type %s is not implemented' % self.sp_type)

        cost = T.zeros((), dtype=floatX)

        params = []
        if self.sp_weight['h']: 
            cost += self.sp_weight['h']  * T.sum(loss(self.sp_targ['h'], hack_h.mean(axis=0)))
            params += [self.hbias]

        if self.sp_type in ['kl'] and self.sp_weight['h']:
            params += [self.Wv, self.alpha_h, self.mu_h]

        return utils_cost.Cost(cost, params)


    ##############################
    # GENERIC OPTIMIZATION STUFF #
    ##############################
    def get_reg_cost(self, l2=None, l1=None):
        """
        Builds the symbolic expression corresponding to first-order gradient descent
        of the cost function ``cost'', with some amount of regularization defined by the other
        parameters.
        :param l2: dict whose values represent amount of L2 regularization to apply to
        parameter specified by key.
        :param l1: idem for l1.
        """
        cost = T.zeros((), dtype=floatX)
        params = []

        for p in self.params():

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

    def get_monitoring_channels(self, x, y=None):
        chans = {}
        chans.update(self.monitor_matrix(self.Wv))
        chans.update(self.monitor_vector(self.hbias))
        chans.update(self.monitor_vector(self.alpha_h_prec, name='alpha_h_prec'))
        chans.update(self.monitor_vector(self.alpha_v_prec, name='alpha_v_prec'))
        chans.update(self.monitor_vector(self.mu_h))
        chans.update(self.monitor_vector(self.mu_v))
        chans.update(self.monitor_matrix(self.neg_h))
        chans.update(self.monitor_matrix(self.neg_s))
        chans.update(self.monitor_matrix(self.neg_v))
        wv_norms = T.sqrt(T.sum(self.Wv**2, axis=0))
        chans['wv_norm.mean'] = T.mean(wv_norms)
        chans['wv_norm.max'] = T.max(wv_norms)
        chans['wv_norm.min'] = T.min(wv_norms)
        chans['lr'] = self.lr
        return chans


class TrainingAlgorithm(default.DefaultTrainingAlgorithm):

    def setup(self, model, dataset):
        super(TrainingAlgorithm, self).setup(model, dataset)
