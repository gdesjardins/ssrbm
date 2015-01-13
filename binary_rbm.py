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
import theano
import theano.tensor as T
from . import bin_ss_rbm

class BinaryRBM(bin_ss_rbm.BinarySpikeSlabRBM):

    def params(self):
        """
        Returns a list of learnt model parameters.
        """
        return [self.Wv, self.vbias, self.hbias]

    def energy(self, h_sample, s_sample, v_sample):
        """
        Computes energy for a given configuration of (g,h,v,x,y).
        :param h_sample: T.matrix of shape (batch_size, n_h)
        :param s_sample: T.matrix of shape (batch_size, n_s)
        :param v_sample: T.matrix of shape (batch_size, n_v)
        """
        energy = -T.sum(s_sample * T.dot(v_sample, self.Wv) * h_sample, axis=1)
        energy -= T.dot(h_sample, self.hbias)
        energy -= T.dot(v_sample, self.vbias)
        return energy

    def free_energy(self, v_sample):
        fe = -T.dot(v_sample, self.vbias)
        h_mean = self.h_given_v_input(v_sample)
        fe -= T.sum(T.nnet.softplus(h_mean), axis=1)
        return fe

    def h_given_v_input(self, v_sample):
        """
        Compute mean activation of h given v.
        :param v_sample: T.matrix of shape (batch_size, n_v matrix)
        """
        h_mean = T.dot(v_sample, self.Wv) + self.hbias
        return h_mean
 
    def s_given_hv(self, h_sample, v_sample):
        return T.ones_like(h_sample)

    def sample_s_given_hv(self, h_sample, v_sample):
        return T.ones_like(h_sample)

    def __call__(self, v):
        h_mean = self.h_given_v(v)
        return h_mean
