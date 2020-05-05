import functools
from typing import Sequence, Union

import numpy as np
import scipy.stats


class ScipyDistribution:
    """
    Class ScipyDistribution is a handy wrap-up around scipy.stats.rv_continuous. It has the same interface as
    scipy.stats.rv_continuous, but without distribution parameters whose values must initially be passed to the
    constructor together with a particular instance of scipy.stats.rv_continuous. Here is an example of usage of
    ScipyDistribution::

        d = ScipyDistribution(scipy.stats.beta, 0.5, 2.)
        first_decile = d.ppf(0.1)

    """
    def __init__(self, rv_obj: scipy.stats.rv_continuous,  *args):
        self._rv_obj = rv_obj
        self._rv_obj_args = args

    def __getattr__(self, item):
        if item in ['rvs', 'stats', 'entropy', 'median', 'mean', 'var', 'std']:
            # for example: std(a, b, loc=0, scale=1)
            return functools.partial(getattr(self._rv_obj, item), *self._rv_obj_args)
        elif item == 'expect':
            return lambda func, **kwargs: self._rv_obj.expect(func, args=tuple(self._rv_obj_args), **kwargs)
        else:
            # for example: pdf(x, a, b, loc=0, scale=1)
            return lambda first_arg, **kwargs: getattr(self._rv_obj, item)(first_arg, *self._rv_obj_args, **kwargs)


class EmpiricalDistribution:
    """
    Class EmpiricalDistribution represents an empirical distribution which is viewed as a discrete distribution
    scipy.stats.rv_discrete (so class EmpiricalDistribution has the same interface as rv_discrete even not
    subclassing it). Using rv_discrete methods, one can calculate any statistic of the empirical distribution (including
    arbitrary expected values).

    Note:
      This class, though representing a discrete distribution, does not subclass rv_discrete. It is all because
      of the issue in rv_discrete constructor (see https://github.com/scipy/scipy/issues/8057 for details).
    
    Warning:
      rv_discrete actually takes only integer values, but we may pass floating numbers instead. It works OK,
      but may be dangerous (see https://github.com/scipy/scipy/issues/3758 for details).
    """
    def __init__(self, data_samples: np.ndarray, **kwargs):
        self.data_samples = np.sort(data_samples)
        self.indices = np.zeros((len(self.data_samples),), dtype=int)
        probs = np.zeros_like(self.indices, dtype=float)
        j = 0  # counter for probs; after the cycle, it will store the number of elements in discrete domain
        probs[0] = 1.
        for i in range(1, len(self.data_samples)):
            if self.data_samples[i] != self.data_samples[i - 1]:
                j += 1
            probs[j] += 1.
            self.indices[j] = i
        probs = probs[:j+1]
        self.indices = self.indices[:j+1]
        probs /= float(len(probs))
        unique_values = np.take(self.data_samples, self.indices)
        values = (unique_values, probs)
        self._rv_obj = scipy.stats.rv_discrete(values=values, **kwargs)

    def __getattr__(self, item):
        return getattr(self._rv_obj, item)

    def histogram_distribution(self, bins=Union[int, Sequence[float]]) -> scipy.stats.rv_histogram:
        """
        Returns histogram distribution based on the empirical distribution

        :param bins: see numpy.histogram docs for the explanation
        :return: scipy.stats.rv_histogram object
        """
        hist = np.histogram(self.data_samples, bins=bins)
        return scipy.stats.rv_histogram(hist)
