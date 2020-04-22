import functools

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
