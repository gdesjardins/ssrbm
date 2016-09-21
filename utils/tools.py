import numpy
from . import floatX

class HyperParamIterator():

    def __init__(self, timestamps, vals):
        assert len(timestamps) != 0
        assert len(timestamps) == len(vals)

        self.i = 0
        self.n = 0

        if not isinstance(vals, (list, numpy.ndarray)):
            vals = [vals]
        if not isinstance(timestamps, (list, numpy.ndarray)):
            timestamps = [timestamps]

        self.vals = vals if isinstance(vals[0], int) else numpy.asarray(vals, dtype=floatX)
        self.timestamps = numpy.asarray(timestamps, dtype=floatX)
        try:
            self.value = self.vals[0]
        except:
            import pdb; pdb.set_trace()

    def next(self):
        rval = False

        if self.i+1 < len(self.timestamps) and \
           self.n >= self.timestamps[self.i+1]:
            self.i += 1
            self.value = self.vals[self.i]
            rval = True

        self.n += 1
        return rval

    def __iterator__(self):
        return self

