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
import md5
import pickle
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import function, shared
from theano.sandbox import linalg

from pylearn2.training_algorithms import default
from pylearn2.utils import serial
from pylearn2.base import Block
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace

from utils import tools
from utils import rbm_utils
from utils import cost as utils_cost
from utils import sharedX, floatX, npy_floatX
import orthogonality
import truncated

class PooledSpikeSlabRBM(Model, Block):
    """Spike & Slab Restricted Boltzmann Machine (RBM)  """

    def load_params(self, model):
        fp = open(model)
        model = pickle.load(fp)
        fp.close()

        self.Wv.set_value(model.Wv.get_value())
        self.hbias.set_value(model.hbias.get_value())
        self.mu.set_value(model.mu.get_value())
        self.alpha.set_value(model.alpha.get_value())
        self.beta.set_value(model.beta.get_value())
        self.wv_norms.set_value(model.wv_norms.get_value())
        # sync negative phase particles
        self.neg_v.set_value(model.neg_v.get_value())
        self.neg_ev.set_value(model.neg_ev.get_value())
        self.neg_s.set_value(model.neg_s.get_value())
        self.neg_h.set_value(model.neg_h.get_value())
        self.sp_pos_v.set_value(model.sp_pos_v.get_value())
        self.sp_pos_h.set_value(model.sp_pos_h.get_value())
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
        self.vbound.set_value(model.vbound.get_value())


    def __init__(self, 
            input=None, Wv=None, vbias=None, hbias=None,
            numpy_rng = None, theano_rng = None,
            n_h=100, bw_s=1, n_v=100, init_from=None,
            neg_sample_steps=1,
            lr=None, lr_timestamp=None, lr_mults = {},
            iscales={}, clip_min={}, clip_max={}, vbound=5.,
            l1 = {}, l2 = {}, orth_lambda=0.,
            var_param_alpha='exp', var_param_beta='linear',
            sp_type='kl', sp_weight={}, sp_targ={},
            batch_size = 13,
            scalar_b = False,
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
        for k in ['Wv', 'vbias', 'hbias']: assert k in iscales.keys()
        iscales.setdefault('mu', 1.)
        iscales.setdefault('alpha', 0.)
        iscales.setdefault('beta', 0.)
        for k in ['h']: assert k in sp_weight.keys()
        for k in ['h']: assert k in sp_targ.keys()

        self.jobman_channel = None
        self.jobman_state = {}
        self.register_names_to_del(['jobman_channel'])

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
        self.n_s = self.n_h * self.bw_s
        
        self.wv_norms = sharedX(1.0 * numpy.ones(self.n_s), name='wv_norms')
        if Wv is None:
            wv_val =  self.rng.randn(n_v, self.n_s) * iscales['Wv']
            self.Wv = sharedX(wv_val, name='Wv')
        else:
            self.Wv = Wv

        self.Wh = numpy.zeros((self.n_s, self.n_h), dtype=floatX)
        for i in xrange(self.n_h):
            self.Wh[i*bw_s:(i+1)*bw_s, i] = 1.

        # allocate shared variables for bias parameters
        if hbias is None:
            self.hbias = sharedX(iscales['hbias'] * numpy.ones(n_h), name='hbias') 
        else:
            self.hbias = hbias

        # mean (mu) and precision (alpha) parameters on s
        self.mu = sharedX(iscales['mu'] * numpy.ones(self.n_s), name='mu')
        self.alpha = sharedX(iscales['alpha'] * numpy.ones(self.n_s), name='alpha')
        var_param_func = {'exp': T.exp,
                          'softplus': T.nnet.softplus,
                          'linear': lambda x: x}
        self.alpha_prec = var_param_func[self.var_param_alpha](self.alpha)

        # diagonal of precision matrix of visible units
        self.vbound = sharedX(vbound, name='vbound')
        self.beta = sharedX(iscales['beta'] * numpy.ones(n_v), name='beta')
        self.beta_prec = var_param_func[self.var_param_beta](self.beta)

        # allocate shared variable for persistent chain
        self.neg_v  = sharedX(self.rng.rand(batch_size, n_v), name='neg_v')
        self.neg_ev = sharedX(self.rng.rand(batch_size, n_v), name='neg_ev')
        self.neg_s  = sharedX(self.rng.rand(batch_size, self.n_s), name='neg_s')
        self.neg_h  = sharedX(self.rng.rand(batch_size, n_h), name='neg_h')
       
        # moving average values for sparsity
        self.sp_pos_v = sharedX(self.rng.rand(1,self.n_v), name='sp_pos_v')
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
        self.input = T.matrix('input') if input is None else input
        
        # configure input-space (new pylearn2 feature?)
        self.input_space = VectorSpace(n_v)
        self.output_space = VectorSpace(n_h)

        self.batches_seen = 0                    # incremented on every batch
        self.examples_seen = 0                   # incremented on every training example
        self.force_batch_size = batch_size  # force minibatch size

        self.error_record = []
 
        if compile: self.do_theano()

        #### load layer 1 parameters from file ####
        if init_from:
            self.load_params(init_from)

    def params(self):
        """
        Returns a list of learnt model parameters.
        """
        params = [self.Wv, self.hbias, self.alpha, self.mu, self.beta]
        if self.flags.get('split_norm', False):
            params += [self.wv_norms]
        return params

    def do_theano(self):
        """ Compiles all theano functions needed to use the model"""

        init_names = dir(self)

        ###### All fields you don't want to get pickled (e.g., theano functions) should be created below this line
        # SAMPLING: NEGATIVE PHASE
        neg_updates = self.neg_sampling_updates(
                n_steps=self.neg_sample_steps,
                use_pcd=not self.flags['use_cd'])
        self.sample_func = theano.function([], [], updates=neg_updates)

        # determing maximum likelihood cost
        if self.flags.get('use_energy', False):
            pos_h = self.h_given_v(self.input)
            pos_s = self.s_given_hv(pos_h, self.input)
            ml_cost = self.ml_cost_energy(pos_h = pos_h, pos_s = pos_s, pos_v = self.input,
                                          neg_h = neg_updates[self.neg_h],
                                          neg_s = neg_updates[self.neg_s],
                                          neg_v = neg_updates[self.neg_v])
        else:
            ml_cost = self.ml_cost_free_energy(pos_v = self.input, neg_v = neg_updates[self.neg_v])

        main_cost = [ml_cost,
                     self.get_sparsity_cost(),
                     self.get_reg_cost(self.l2, self.l1)]
        if self.orth_lambda:
            main_cost += [self.orthogonality_cost(self.orth_lambda)]

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
        self.batch_train_func = function([self.input], [],
                                         updates=learning_updates,
                                         name='train_rbm_func')


        # enforce constraints function
        constraint_updates = OrderedDict() 
        constraint_updates[self.wv_norms] = T.maximum(1.0, self.wv_norms)
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
        if self.scalar_b:
            beta = constraint_updates.get(self.beta, self.beta)
            constraint_updates[self.beta] = T.mean(beta) * T.ones_like(beta)
        self.enforce_constraints = theano.function([],[], updates=constraint_updates)

        ###### All fields you don't want to get pickled should be created above this line
        final_names = dir(self)
        self.register_names_to_del( [ name for name in (final_names) if name not in init_names ])

        # Before we start learning, make sure constraints are enforced
        self.enforce_constraints()

    def train_batch(self, dataset, batch_size):

        if self.batches_seen == 0:
            if self.vbound.get_value() == 0:
                self.vbound.set_value(abs(dataset.X).max())
            print 'Truncating visibles to +/- %f' % self.vbound.get_value()

        x = dataset.get_batch_design(batch_size, include_labels=False)
        if self.flags.get('truncated_normal', False):
            vbound = self.vbound.get_value()
            x = numpy.clip(x, -vbound, vbound)
        self.batch_train_func(x)

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

    def energy(self, h_sample, s_sample, v_sample):
        """
        Computes energy for a given configuration of (g,h,v,x,y).
        :param h_sample: T.matrix of shape (batch_size, n_h)
        :param s_sample: T.matrix of shape (batch_size, n_s)
        :param v_sample: T.matrix of shape (batch_size, n_v)
        """
        Wv = self.wv_norms * self.Wv
        from_v = T.dot(v_sample, Wv)
        from_h = T.dot(h_sample, self.Wh.T)
        energy = -T.sum(s_sample * from_v * from_h, axis=1)
        energy += T.sum(0.5 * self.alpha_prec * s_sample**2, axis=1)
        energy -= T.sum(self.alpha_prec * self.mu * s_sample * from_h, axis=1)
        energy += T.sum(0.5 * self.alpha_prec * self.mu**2 * from_h, axis=1)
        energy += T.sum(0.5 * self.beta_prec * v_sample**2, axis=1)
        energy -= T.dot(h_sample, self.hbias)
        return energy

    def free_energy(self, v_sample):
        fe  = T.sum(0.5 * self.beta_prec * v_sample**2, axis=1)
        fe -= 0.5 * T.sum(T.log(2*numpy.pi / self.alpha_prec))
        h_mean = self.h_given_v_input(v_sample)
        fe -= T.sum(T.nnet.softplus(h_mean), axis=1)
        return fe

    def __call__(self, v, output_type='h'):
        assert output_type in ['h', 'hs']
        h_mean = self.h_given_v(v)
        s_mean = self.s_given_hv(h_mean, v)
        output_prods = {
                'h': h_mean,
                'hs': h_mean * T.sqrt(T.dot(s_mean**2, self.Wh))
                }
        return output_prods[output_type]

    ######################################
    # MATH FOR CONDITIONAL DISTRIBUTIONS #
    ######################################

    def h_given_v_input(self, v_sample):
        """
        Compute mean activation of h given v.
        :param v_sample: T.matrix of shape (batch_size, n_v matrix)
        """
        Wv = self.wv_norms * self.Wv
        from_v = T.dot(v_sample, Wv)
        s_mean  = 0.5 * 1./self.alpha_prec * from_v**2
        s_mean += from_v * self.mu
        h_mean = T.dot(s_mean, self.Wh) + self.hbias
        return h_mean
    
    def h_given_v(self, v_sample):
        h_mean = self.h_given_v_input(v_sample)
        return T.nnet.sigmoid(h_mean)

    def sample_h_given_v(self, v_sample, rng=None):
        """
        Generates sample from p(h|v)
        """
        h_mean = self.h_given_v(v_sample)

        rng = self.theano_rng if rng is None else rng
        h_sample = rng.binomial(size=(self.batch_size,self.n_h),
                                            n=1, p=h_mean, dtype=floatX)
        return h_sample

    def s_given_hv(self, h_sample, v_sample):
        Wv = self.wv_norms * self.Wv
        from_v = T.dot(v_sample, Wv)
        from_h = T.dot(h_sample, self.Wh.T)
        s_mean = (1./self.alpha_prec * from_v + self.mu) * from_h
        return s_mean

    def sample_s_given_hv(self, h_sample, v_sample, rng=None):
        s_mean = self.s_given_hv(h_sample, v_sample)
        
        rng = self.theano_rng if rng is None else rng
        s_sample = rng.normal(
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
        Wv = self.wv_norms * self.Wv
        from_h = T.dot(h_sample, self.Wh.T)
        v_mean = 1./self.beta_prec * T.dot(s_sample * from_h, Wv.T)
        return v_mean

    def sample_v_given_hs(self, h_sample, s_sample, rng=None):
        v_mean = self.v_given_hs(h_sample, s_sample)

        rng = self.theano_rng if rng is None else rng
        if self.flags.get('truncated_normal', False):
            v_sample = truncated.truncated_normal(
                    size=(self.batch_size, self.n_v),
                    avg = v_mean, 
                    std = T.sqrt(1./self.beta_prec),
                    lbound = -self.vbound,
                    ubound = self.vbound,
                    theano_rng = rng,
                    dtype=floatX)
        else:
            v_sample = rng.normal(
                    size=(self.batch_size, self.n_v),
                    avg = v_mean, 
                    std = T.sqrt(1./self.beta_prec), dtype=floatX)
        return v_sample

    def numpy_compute_covariance_v_h(self):
        Wv = self.Wv.get_value()
        mu = self.mu.get_value()
        alpha = numpy.exp(self.alpha.get_value())
        beta = numpy.exp(self.beta.get_value())
        h_sample = self.neg_h.get_value()
        from_h = numpy.dot(h_sample, self.Wh.T)
        cov = []
        for i in xrange(self.batch_size):
            print 'Computing covariance for %i-th sample...' % i
            temp = 1./alpha * from_h[i] * Wv
            prec = numpy.diag(beta) - numpy.dot(temp, Wv.T)
            cov += [numpy.linalg.inv(prec)]
            if numpy.isnan(cov[-1]).any():
                import pdb; pdb.set_trace()
        return cov

    def do_debug(self):
        rng = RandomStreams(seed=12312)
        in1 = T.matrix()
        in2 = T.matrix()
        self.h_given_v_func = theano.function([in1], self.h_given_v(in1))
        self.sample_h_given_v_func = theano.function([in1], self.sample_h_given_v(in1, rng=rng))
        self.s_given_hv_func  = theano.function([in1, in2], self.s_given_hv(in1, in2))
        self.sample_s_given_hv_func = theano.function([in1, in2], self.sample_s_given_hv(in1, in2, rng=rng))
        self.v_given_hs_func = theano.function([in1, in2], self.v_given_hs(in1, in2))
        self.sample_v_given_hs_func = theano.function([in1, in2], self.sample_v_given_hs(in1, in2, rng=rng))

    ##################
    # SAMPLING STUFF #
    ##################

    def neg_sampling(self, v_sample, n_steps=1):
        """
        Gibbs step for negative phase, which alternates: 
        p(h|v), p(s|h,v) and p(v|h,s)
        :param v_sample: T.matrix of shape (batch_size, n_v)
        :param n_steps: number of Gibbs updates to perform in negative phase.
        """

        def gibbs_iteration(v1):
            h2 = self.sample_h_given_v(v1)
            s2 = self.sample_s_given_hv(h2, v1)
            v2 = self.sample_v_given_hs(h2, s2)
            return [h2, s2, v2]

        [new_h, new_s, new_v] , updates = theano.scan(
                gibbs_iteration,
                outputs_info = [None, None, v_sample],
                n_steps=n_steps)

        return [new_h[-1], new_s[-1], new_v[-1]]

    def neg_sampling_updates(self, n_steps=1, use_pcd=True):
        """
        Implements the negative phase, generating samples from p(h,s,v).
        :param n_steps: scalar, number of Gibbs steps to perform.
        """
        init_chain = self.neg_v if use_pcd else self.input
        [new_h, new_s, new_v] =  self.neg_sampling(init_chain, n_steps=n_steps)

        # we want to plot the expected value of the samples
        new_ev = self.v_given_hs(new_h, new_s)

        updates = OrderedDict()
        updates[self.neg_h] = new_h
        updates[self.neg_s] = new_s
        updates[self.neg_v] = new_v
        updates[self.neg_ev] = new_ev

        return updates

    def ml_cost_free_energy(self, pos_v, neg_v):
        pos_cost = T.sum(self.free_energy(pos_v))
        neg_cost = T.sum(self.free_energy(neg_v))
        batch_cost = pos_cost - neg_cost
        cost = batch_cost / self.batch_size
        return utils_cost.Cost(cost, self.params(), [pos_v,neg_v])

    def ml_cost_energy(self, pos_h, pos_s, pos_v, neg_h, neg_s, neg_v):
        pos_cost = T.sum(self.energy(pos_h, pos_s, pos_v))
        neg_cost = T.sum(self.energy(neg_h, neg_s, neg_v))
        batch_cost = pos_cost - neg_cost
        cost = batch_cost / self.batch_size
        return utils_cost.Cost(cost, self.params(),
                [pos_h, pos_s, pos_v, neg_h, neg_s, neg_v])

    def orthogonality_cost(self, orth_lambda):
        Wv = self.wv_norms * self.Wv
        cost = orthogonality.orthogonal_pools(Wv, self.bw_s)
        params = [self.Wv]
        if self.flags.get('split_norm', False):
            params += [self.wv_norms]
        return utils_cost.Cost(orth_lambda * cost, params)

    def get_sparsity_cost(self):

        # update mean activation using exponential moving average
        hack_h = self.h_given_v(self.sp_pos_v)

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
            params += [self.Wv, self.alpha, self.mu]
            if self.flags.get('split_norm', False):
                params += [self.wv_norms]

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
        chans.update(self.monitor_vector(self.wv_norms))
        chans.update(self.monitor_vector(self.hbias))
        chans.update(self.monitor_vector(self.alpha_prec, name='alpha_prec'))
        chans.update(self.monitor_vector(self.mu))
        chans.update(self.monitor_vector(self.beta_prec, name='beta_prec'))
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
