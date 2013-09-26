'''
Created on Mar 20, 2013

@author: ejhumphrey
'''
import os
import theano.tensor as T
from . import FLOATX
import theano

# TODO: just taking the mean over a batch is not insignificant... there are 
# definitely other ways to do this, geometric mean (ignore outliers, 
# arithmetic-AND), max (cater to big error, arithmetic-OR). This is 
# particularly interesting, because simple averages aren't always 
# what you want...

def negative_log_likelihood(x_input):
    """
    Returns
    -------
    scalar_loss : symbolic scalar
        Cost of this penalty
    inputs : dict
        Dictionary of full param names and symbolic parameters.
    """
    y_target = T.ivector('y_target')
    scalar_loss = T.mean(-T.log(x_input)[T.arange(y_target.shape[0],
                                                  dtype='int32'),
                                         y_target])
    return scalar_loss, [y_target]

def l2_penalty(x_input):
    """
    Returns
    -------
    scalar_loss : symbolic scalar
        Cost of this penalty
    inputs : dict
        Dictionary of full param names and symbolic parameters.
    """
    hyperparam_name = os.path.join(x_input.name, 'l2_penalty')
    weight_decay = T.scalar(hyperparam_name, dtype=FLOATX)
    scalar_loss = weight_decay * T.sum(T.pow(x_input, 2.0))
    return scalar_loss, [weight_decay]

def l1_penalty(x_input):
    """
    Returns
    -------
    scalar_loss : symbolic scalar
        Cost of this penalty
    inputs : dict
        Dictionary of full param names and symbolic parameters.
    """
    hyperparam_name = os.path.join(x_input.name, 'l1_penalty')
    sparsity = T.scalar(hyperparam_name, dtype=FLOATX)
    scalar_loss = sparsity * T.sum(T.abs_(x_input))
    return scalar_loss, [sparsity]

def maximum_likelihood_scorer(x_input, target_idx):
    return T.mean(T.eq(T.argmax(x_input, axis=1), target_idx))

def maximum_likelihood_error(x_input, target_idx):
    return 1.0 - maximum_likelihood_scorer(x_input, target_idx)

def euclidean(a, b):
    a, b = a.flatten(2), b.flatten(2)
    return T.sqrt(T.sum(T.pow(a - b, 2.0), axis=1))

def euclidean_loss(a, b):
    """
    Returns
    -------
    scalar_loss : symbolic scalar
        Cost of this penalty
    inputs : dict
        Dictionary of full param names and symbolic parameters.
    """
    scalar_loss = T.mean(euclidean(a, b))
    return scalar_loss, []

def manhattan(a, b):
    a, b = a.flatten(2), b.flatten(2)
    return T.sum(T.abs_(a - b), axis=1)

def manhattan_loss(a, b):
    """
    Returns
    -------
    scalar_loss : symbolic scalar
        Cost of this penalty
    inputs : dict
        Dictionary of full param names and symbolic parameters.
    """
    scalar_loss = T.mean(manhattan(a, b))
    return scalar_loss, []

LossFunctions = {"nll":negative_log_likelihood,
                 "l2_penalty":l2_penalty,
                 "l1_penalty":l1_penalty}

class Loss(object):
    def __init__(self):
        self._inputs = list()
        self._total = 0.0

    def register(self, graph, input_name, loss_type):
        """
        Parameters
        ----------
        graph : Instance of a dnn.core.graph
        loss_pairs : tuple
            Pair of strings pointing to a symbolic variable in graph.vars and a
            registered loss function in LossFunctions.

        """
        for x in graph.inputs:
            if not x in self._inputs:
                self._inputs.append(x)

        fx = LossFunctions.get(loss_type)
        assert input_name in graph.vars, \
            "No variable named '%s' in vars: %s" % (input_name,
                                                    self.vars.keys())
        loss_input = graph.vars.get(input_name)
        scalar_loss, extra_inputs = fx(loss_input)
        self._total += scalar_loss
        self._inputs.extend(extra_inputs)

    @property
    def total(self):
        return self._total

    @property
    def inputs(self):
        return list(self._inputs)

    def empty_inputs(self, fill_value=0.0):
        return dict([(x.name, fill_value) for x in self.inputs])

    def compile(self):
        self._fx = theano.function(inputs=self.inputs,
                                   outputs=self.total,
                                   allow_input_downcast=True,
                                   on_unused_input='warn')

    def __call__(self, inputs):
        """
        """
        if self._fx is None:
            self.compile()
        return self._fx(**inputs)

'''
class CrossEntropy(LossFunc):
    """
    Ack! I'm an empty docstring
    """
    def __init__(self, bounds=(-1, 1), *args, **kwargs):
        LossFunc.__init__(self, *args, **kwargs)
        self.bounds = bounds

    def cost(self, a, b):
        """
        a : input
        b : target

        make a more like b, the other way around breaks
        """
        a = (a - self.bounds[0]) / float(self.bounds[1] - self.bounds[0])
        b = (b - self.bounds[0]) / float(self.bounds[1] - self.bounds[0])
        return -T.sum(b * T.log(a) + (1.0 - b) * T.log(1.0 - a), axis=1)

class CrossEntropyTarget(CrossEntropy):
    """
    Convenience around CrossEntropy to generate its own target
    """
    def __init__(self, bounds=(0, 1), *args, **kwargs):
        CrossEntropy.__init__(self, bounds=bounds, *args, **kwargs)
        self.y = T.matrix('y', dtype=FLOATX)
        self._inputs += [self.y]

    def cost(self, a):
        return CrossEntropy.cost(self, a, self.y)

class CrossEntropyTargetMask(CrossEntropy):
    """
    Convenience around CrossEntropy to generate its own target
    """
    def __init__(self, bounds=(0, 1), *args, **kwargs):
        CrossEntropy.__init__(self, bounds=bounds, *args, **kwargs)
        self.y = T.matrix('y', dtype=FLOATX)
        self.mask = T.matrix('mask', dtype=FLOATX)
        self._inputs += [self.y, self.mask]

    def cost(self, a):
        """
        a : input
        b : target

        make a more like b, the other way around breaks
        """
        a = (a - self.bounds[0]) / float(self.bounds[1] - self.bounds[0])
        b = (self.y - self.bounds[0]) / float(self.bounds[1] - self.bounds[0])
        x = b * T.log(a) + (1.0 - b) * T.log(1.0 - a)
        return -T.sum(x * self.mask, axis=1)

class CDMargin(LossFunc):
    """
    Should modify this to make the hinge a Theano scalar
    """
    def __init__(self, m=1.25, **kwargs):
        self.m = m
        LossFunc.__init__(self, **kwargs)
        self.y = T.vector('y')
        self._inputs += [self.y]

    def cost(self, x):
        """
        Note that
            x is a vector distance.
            y==1==True==Same
        """
        diff_loss = 0.5 * (1.0 - self.y) * self._diff_cost(x)
        same_loss = 0.5 * self.y * self._same_cost(x)
        return diff_loss + same_loss

    def _same_cost(self, x):
        return T.pow(x, 2.0)

    def _diff_cost(self, x):
        return T.pow(soft_hinge(self.m, x), 2.0)


class NLLLogLoss(LossFunc):
    """
    Should modify this to make the hinge a Theano scalar
    """
    def __init__(self, m=0.0, **kwargs):
        LossFunc.__init__(self, **kwargs)
        self.m = m
        self.y_idx = T.vector('y_idx', dtype='int32')
        self.y_valid = T.matrix('y_correct', dtype='float32')
        self._inputs += [self.y_valid, self.y_idx]

    def cost(self, x):
        """

        """
        # total energy
        energy = -T.log(x)
        #slice the energy of the indices of interest, now a vector
        e_i = energy[T.arange(self.y_idx.shape[0], dtype='int32'), self.y_idx]
        # Add an offset to the right answers
        energy_moia = energy + self.y_valid * T.max(energy, axis=1).dimshuffle(0, 'x')
        e_moia = energy_moia.min(axis=1)
        return soft_hinge(e_i - e_moia, self.m)

class CDLogLoss(LossFunc):
    """
    Should modify this to make the hinge a Theano scalar
    """
    def __init__(self, m=1.25, **kwargs):
        LossFunc.__init__(self, **kwargs)
        self.m = m
        self.y_idx = T.vector('y_idx', dtype='int32')
        self.y_valid = T.matrix('y_correct', dtype='float32')
        self._inputs += [self.y_valid, self.y_idx]

    def cost(self, x):
        """

        """
        # total energy
        energy = -T.log(x)
        #slice the energy of the indices of interest, now a vector
        e_i = energy[T.arange(self.y_idx.shape[0], dtype='int32'), self.y_idx]
        # Add an offset to the right answers
        energy_moia = energy + self.y_valid * T.max(energy, axis=1).dimshuffle(0, 'x')
        e_moia = energy_moia.min(axis=1)
        return e_i + soft_hinge(self.m, e_moia)


# --- update monitor ---
class Monitor(object):
    def __init__(self, max_iter=10000, max_time=28800, **kwargs):
        self.max_iter = max_iter
        self.max_time = max_time

        self.tstamps = []
        self.iter_total = 0
        self.loss_queue = []
        self.stats = {'Time Elapsed':_timefmt(0.0),
                      'Time Remaining':_timefmt(max_time),
                      'Error':None,
                      'Loss':None}
        self.reset_clock()
        self.report_buffer = []
        self.header = {'dataset':"%s" % kwargs.get('dset', 'None'),
                       'runtime':time.asctime(),
                       'model':"%s" % kwargs.get('model', ''), }

    def reset_clock(self):
        self.start_time = time.time()
        self.time_elapsed = 0.0
        self.iter_since_last_reset = 0

    # TODO: Perhaps, like timeit, dynamically figure out how often to
    #       print updates based on how long the average iteration takes.

    def iter(self):
        return self.iter_total

    def update(self, loss, **kwargs):
        for k in kwargs:
            if not k in ['error']:
                raise ValueError("invalid keyword: %s" % k)
            if k == 'error':
                self.stats['Error'] = "%0.3f" % kwargs.get('error')

        self.iter_total += 1
        self.iter_since_last_reset += 1

        self.loss_queue.append(loss)
        self.tstamps += [time.time()]
        self.time_elapsed = self.tstamps[-1] - self.start_time

        self.stats['Loss'] = "%0.3f" % loss
        self.stats['Iteration'] = self.iter_total
        self.stats['Time Elapsed'] = _timefmt(self.time_elapsed)
        self.stats['Time Remaining'] = _timefmt((self.time_elapsed / float(self.iter_since_last_reset)) \
                                                      * (self.max_iter - self.iter_total))

    def __str__(self, *args, **kwargs):
        to_print = {}
        for k in self.stats:
            if not self.stats.get(k) is None:
                to_print[k] = self.stats.get(k)
        return _pprint(to_print, offset=4, printer=str)

    def update_buffer(self):
        self.report_buffer += ["%s" % self]

    def write_report(self, fout):
        fh = open(fout, 'w')
        hdr = "".join(["%s: %s\t" % (k, self.header[k]) for k in self.header])
        [fh.write(l + "\n") for l in [hdr] + self.report_buffer]
        fh.close()

    def finished(self):
        """
        subclass for different early stopping conditions
        """
        if self.iter_total >= self.max_iter:
            return True
        elif self.time_elapsed >= self.max_time:
            return True

        return False
'''
