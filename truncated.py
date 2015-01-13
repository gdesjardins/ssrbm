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
from theano.printing import Print
floatX = theano.config.floatX

SQRT2 = numpy.cast[floatX](numpy.sqrt(2))
def truncated_normal(size, avg, std, lbound, ubound, theano_rng, dtype):

    def phi(x):
        erfarg = (x - avg) / (std * SQRT2)
        rval = 0.5 * (1. + T.erf(erfarg))
        return rval.astype(dtype)
    
    def phi_inv(phi_x):
        erfinv_input = T.clip(2. * phi_x - 1., -1.+1e-6, 1.-1e-6)
        rval = avg + std * SQRT2 * T.erfinv(erfinv_input)
        return rval.astype(dtype)

    # center lower and upper bounds based on mean
    u = theano_rng.uniform(size=size, dtype=dtype)

    cdf_range = phi(ubound) - phi(lbound)

    # if avg >> ubound, return ubound
    # if avg << lbound, return lbound
    # else return phi(lbound) + u * [phi(ubound) - phi(lbound)]
    rval = T.switch(cdf_range,
            phi_inv(phi(lbound) + u * cdf_range),
            T.switch(avg >= ubound, ubound, lbound))

    return rval
