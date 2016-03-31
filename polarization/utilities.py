"""
.. module:: utilities
    :platform: Windows
    :synopsis: Implementation of various functions that ease the work,
     but do not belong in one of the other modules.

.. moduleauthor:: Wouter Gins <wouter.gins@fys.kuleuven.be>
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

c = 299792458.0
h = 6.62606957 * (10 ** -34)
q = 1.60217657 * (10 ** -19)

cmap = mpl.colors.ListedColormap(['#A6CEE3', '#1F78B4', '#B2DF8A'])
invcmap = mpl.colors.ListedColormap(['#B2DF8A', '#1F78B4', '#A6CEE3'])

# __all__ = ['ReleaseCurve',
#            'Level',
#            'Energy',
#            'round_to_signif_figs',
#            'weighted_average',
#            'bootstrap_ci',
#            'generate_likelihood_plot',
#            'generate_correlation_plot',
#            'generate_spectrum',
#            'concat_results',
#            'poisson_interval']


def state_number_enumerate(dims, state=None, idx=0):
    """Create the indices for the different entries in
    a multi-dimensional array. Code copied from the QuTiP package.

    Parameters
    ----------
    shape: tuple
        Describes the shape of the multi-dimensional array.

    Returns
    -------
    tuple
        Tuple with each entry being a tuple containing the indices."""
    if state is None:
        state = np.zeros(len(dims))

    if idx == len(dims):
        yield tuple(state)
    else:
        for n in range(dims[idx]):
            state[idx] = n
            for s in state_number_enumerate(dims, state, idx + 1):
                yield s


# Create a vectorized function for calling an array of callables,
# mixed with non-callables.
def ifCallableCall(ob, arg):
    return ob(arg) if callable(ob) else ob

vFifCallableCall = np.vectorize(ifCallableCall, otypes=[np.float])


def callNDArray(arr, arg):
    """Goes over each subarray in the first dimension,
    and calls the corresponding argument. Returns the values itself
    if the entry is not callable.

    Parameters
    ----------
    arr: NumPy array
        Array containing a mix of callable and not-callable entries.
    arg: misc
        Argument to be passed to each callable entry.

    Returns
    -------
    NumPy array
        Array with the callable entries replaced by the returned value."""
    n = arr.shape[0]
    assert n == len(arg)
    res = np.zeros(arr.shape)
    for i in range(n):
        # Go for the vectorized function. In case of problems,
        # comment the following line and use the try-except
        # block. That is proven to work, but is slower.
        res[i] = vFifCallableCall(arr[i], arg[i])
        # try:
        #     res[i] = np.array(
        #         [[x(arg[i]) if callable(x) else x for x in xarr]
        #         for xarr in arr[i]])
        # except ValueError:
        #     raise ValueError()
    return res


class ReleaseCurve(object):

    r"""Creates a callable object for the standard release curve. Formula
    based on J.P. Ramos et al. :cite:`Ramos2014`. Input parameters are
    initialized to an 35Ar release curve.

    Parameters
    ----------
    amp : float,
        Influences the height of the curve, roughly the maximum of the
        release rate. Is also an attribute. Default: 4E7
    a : float between 0 and 1
        Weighting of the different exponentials in the formula. Is also an
        attribute. Default: 0.9
    tr : float
        Time constant parameter in seconds. The attribute is saved as the
        corresponding l-parameter. Default: 78 ms
    tf : float
        Time constant parameter in seconds. The attribute is saved as the
        corresponding l-parameter. Default: 396 ms
    ts : float
        Time constant parameter in seconds. The attribute is saved as the
        corresponding l-parameter. Default: 1905 ms
    pulses : integer
        Number of pulses seperated by the delay parameter. Has no effect if the
        :attr:`continued` parameter is True. Is also an attribute. Default: 3
    delay : float
        Seconds between pulses. Is also an attribute. Default: 10.0 s
    continued : bool
        Continuously generate pulses seperated by the delay parameter if True,
        else create the number of pulses given in the pulses parameter. Is also
        an attribute. Default: True

    Note
    ----
    The l-parameters are related to the t-parameters through
    :math:`l = \frac{\ln(2)}{t}`. The release curve is modeled as:

    .. math::
        RC\left(t\right) = a\left(1-\exp\left(-l_rt\right)\right)
        \left(a\exp\left(-l_ft\right)+(1-a)\exp\left(-l_st\right)\right)"""

    def __init__(self, amp=4.0 * 10 ** 7, a=0.9,
                 tr=78 * (10 ** -3), tf=396 * (10 ** -3),
                 ts=1905 * (10 ** -3),
                 pulses=3, delay=10.0, continued=True):
        super(ReleaseCurve, self).__init__()
        self.amp = amp
        self.a = a

        self.lr = np.log(2) / tr
        self.lf = np.log(2) / tf
        self.ls = np.log(2) / ts

        self.pulses = pulses
        self.delay = delay
        self.continued = continued

    def fit_to_data(self, t, y, yerr):
        """If a release curve is measured as a function of time, this should
        fit the parameters to the given curve y(t) with errors yerr.

        Parameters
        ----------
        t: array_like
            Timevector of the measurements.
        y: array_like
            Counts corresponding to t.
        yerr: array_like
            Counting errors of y.

        Warning
        -------
        This method has not been tested!"""
        import lmfit as lm
        params = lm.Parameters()
        params.add_many(
            ('Amp', self.amp, True, 0, None, None),
            ('a', self.a, True, 0, 1, None, None),
            ('tr', np.log(2) / self.lr, True, None, None, None),
            ('tf', np.log(2) / self.lf, True, None, None, None),
            ('ts', np.log(2) / self.ls, True, None, None, None))

        def resid(params):
            self.amp = params['Amp']
            self.a = params['a']
            self.lr = np.log(2) / params['tr']
            self.lf = np.log(2) / params['tf']
            self.ls = np.log(2) / params['ts']
            return (y - self.empirical_formula(t)) / yerr

        return lm.minimize(resid, params)

    @property
    def pulses(self):
        return self._pulses

    @pulses.setter
    def pulses(self, value):
        self._pulses = int(value)

    @property
    def continued(self):
        return self._continued

    @continued.setter
    def continued(self, value):
        self._continued = (value == 1)

    def empirical_formula(self, t):
        amp = self.amp
        a = self.a
        lr = self.lr
        lf = self.lf
        ls = self.ls

        val = amp * (1 - np.exp(-lr * t)) * (a * np.exp(-lf * t) +
                                             (1 - a) * np.exp(-ls * t))
        return val

    def __call__(self, t):
        """Return the evaluation of the formula, taking the pulses
        and delays into account.

        Parameters
        ----------
        t: array_like
            Times for which the yield is requested."""
        pulses = self.pulses
        delay = self.delay
        continued = self.continued

        pulses = np.arange(1.0, pulses) * delay
        rc = self.empirical_formula(t)
        if not continued:
            for pulsetime in pulses:
                mask = t > pulsetime
                try:
                    if any(mask):
                        rc[mask] += self.empirical_formula(t[mask] - pulsetime)
                except TypeError:
                    if mask:
                        rc += self.empirical_formula(t - pulsetime)
        else:
            pulsetime = delay
            try:
                number = (t // pulsetime).astype('int')
                for pulses in range(1, max(number) + 1):
                    mask = (number >= pulses)
                    rc[mask] += self.empirical_formula(t[mask] -
                                                       pulses * pulsetime)
            except AttributeError:
                number = int(t // pulsetime)
                if number > 0:
                    for i in range(number):
                        rc += self.empirical_formula(t - (i + 1) * pulsetime)
        return rc


class Level(object):

    """Ease-of-use class for representing a level.

    Parameters
    ----------
    energy : float
        Fine structure energy in eV.
    hyp_par : list of 2 floats
        Hyperfine parameters [A, B] in MHz.
    L, S, J : integer or half-integers
        Spin quantum numbers."""

    def __init__(self, energy, hyp_par, L, S, J):
        super(Level, self).__init__()
        self.energy = energy
        self.A, self.B = hyp_par
        self.L = L
        self.S = S
        self.J = J

    def __str__(self):
        s = '<Level object: E=%f, A=%f, B=%f, L=%f, S=%f, J=%f>' % (
            self.energy, self.A, self.B, self.L, self.S, self.J)
        return s


def invCM2MHz(invCM):
    return invCM * 100.0 * c * 10 ** -6


def MHz2invCM(MHz):
    return MHz * 10 ** 6 / (100.0 * c)


def invCM2eV(invCM):
    return invCM * 100.0 * h * c / q


def eV2invCM(eV):
    return eV * q / (100.0 * h * c)


def invCM2nm(invCM):
    return ((invCM * 100.0) ** -1) * (10 ** 9)


def nm2invCM(nm):
    return ((nm * (10 ** -9)) ** -1) / 100.0


class Energy(object):

    """Ease-of-use class to represent energy and frequencies.
    Uses automatic conversion to a series of units.

    Parameters
    ----------
    value: float
        Value of the energy or frequency to be converted/worked with.
    unit: string, {cm-1, MHz, eV, nm}
        String denoting the unit for the given value. Default value is inverse
        centimeters (cm-1)."""

    __units__ = ['cm-1', 'MHz', 'eV', 'nm']
    __conversion__ = {'MHz': invCM2MHz,
                      'eV': invCM2eV,
                      'nm': invCM2nm}

    def __init__(self, value, unit='cm-1'):
        super(Energy, self).__init__()
        if unit not in self.__units__:
            m = '{} is an unknown unit!'.format(unit)
            raise TypeError(m)
        self.unit = unit
        self.value = value
        convert = {'MHz': MHz2invCM,
                   'eV': eV2invCM,
                   'nm': nm2invCM}
        if self.unit in convert.keys():
            self.value = convert[self.unit](self.value)
            self.unit = 'cm-1'

    def __call__(self, unit):
        """Convert the value to the given unit.

        Parameters
        ----------
        unit: string
            Requested unit, must be 'cm-1', 'MHz', 'eV' or 'nm'.

        Returns
        -------
        float
            Converted value."""
        if unit in self.__conversion__.keys():
            val = self.__conversion__[unit](self.value)
        else:
            val = self.value
        return val


def round_to_signif_figs(vals, n):
    """
    Code copied from
    http://stackoverflow.com/questions/18915378/rounding-to-significant-figures-in-numpy
    Goes over the list or array of vals given, and rounds
    them to the number of significant digits (n) given.

    Parameters
    ----------
    vals : array_like
        Values to be rounded.
    n : integer
        Number of significant digits to round to.

    Note
    ----
    Does not accept: inf, nan, complex

    Example
    -------
    >>> m = [0.0, -1.2366e22, 1.2544444e-15, 0.001222]
    >>> round2SignifFigs(m,2)
    array([  0.00e+00,  -1.24e+22,   1.25e-15,   1.22e-03])
    """
    if np.all(np.isfinite(vals)) and np.all(np.isreal((vals))):
        eset = np.seterr(all='ignore')
        mags = 10.0 ** np.floor(np.log10(np.abs(vals)))  # omag's
        vals = np.around(vals / mags, n - 1) * mags  # round(val/omag)*omag
        np.seterr(**eset)
        vals[np.where(np.isnan(vals))] = 0.0  # 0.0 -> nan -> 0.0
    else:
        raise IOError('Input must be real and finite')
    return vals


def weighted_average(x, sigma, axis=None):
    r"""Takes the weighted average of an array of values and the associated
    errors. Calculates the scatter and statistical error, and returns
    the greater of these two values.

    Parameters
    ----------
    x: array_like
        Array-like assortment of measured values, is transformed into a
        1D-array.
    sigma: array_like
        Array-like assortment of errors on the measured values, is transformed
        into a 1D-array.

    Returns
    -------
    tuple
        Returns a tuple (weighted_average, uncertainty), with the uncertainty
        being the greater of the uncertainty calculated from the statistical
        uncertainty and the scattering uncertainty.

    Note
    ----
    The formulas used are

    .. math::

        \left\langle x\right\rangle_{weighted} &= \frac{\sum_{i=1}^N \frac{x_i}
                                                                 {\sigma_i^2}}
                                                      {\sum_{i=1}^N \frac{1}
                                                                {\sigma_i^2}}

        \sigma_{stat}^2 &= \frac{1}{\sum_{i=1}^N \frac{1}{\sigma_i^2}}

        \sigma_{scatter}^2 &= \frac{\sum_{i=1}^N \left(\frac{x_i-\left\langle
                                                    x\right\rangle_{weighted}}
                                                      {\sigma_i}\right)^2}
               {\left(N-1\right)\sum_{i=1}^N \frac{1}{\sigma_i^2}}"""
    # x = np.ravel(x)
    # sigma = np.ravel(sigma)
    Xstat = (1 / sigma**2).sum(axis=axis)
    Xm = (x / sigma**2).sum(axis=axis) / Xstat
    # Xscatt = (((x - Xm) / sigma)**2).sum() / ((1 - 1.0 / len(x)) * Xstat)
    Xscatt = (((x - Xm) / sigma)**2).sum(axis=axis) / ((len(x) - 1) * Xstat)
    Xstat = 1 / Xstat
    return Xm, np.maximum.reduce([Xstat, Xscatt], axis=axis) ** 0.5


def bootstrap_ci(dataframe, kind='basic'):
    """Generate confidence intervals on the 1-sigma level for bootstrapped data
    given in a DataFrame.

    Parameters
    ----------
    dataframe: DataFrame
        DataFrame with the results of each bootstrap fit on a row. If the
        t-method is to be used, a Panel is required, with the data in
        the panel labeled 'data' and the uncertainties labeled 'stderr'
    kind: str, optional
        Selects which method to use: percentile, basic, or t-method (student).

    Returns
    -------
    DataFrame
        Dataframe containing the left and right limits for each column as rows.
"""
    if isinstance(dataframe, pd.Panel):
        data = dataframe['data']
        stderrs = dataframe['stderr']
        args = (data, stderrs)
    else:
        data = dataframe
        args = (data)

    def percentile(data, stderrs=None):
        CI = pd.DataFrame(index=['left', 'right'], columns=data.columns)
        left = data.apply(lambda col: np.percentile(col, 15.865), axis=0)
        right = data.apply(lambda col: np.percentile(col, 84.135), axis=0)
        CI.loc['left'] = left
        CI.loc['right'] = right
        return CI

    def basic(data, stderrs=None):
        CI = pd.DataFrame(index=['left', 'right'], columns=data.columns)
        left = data.apply(lambda col: 2 * col[0] - np.percentile(col[1:],
                                                                 84.135),
                          axis=0)
        right = data.apply(lambda col: 2 * col[0] - np.percentile(col[1:],
                                                                  15.865),
                           axis=0)
        CI.loc['left'] = left
        CI.loc['right'] = right
        return CI

    def student(data, stderrs=None):
        CI = pd.DataFrame(index=['left', 'right'], columns=data.columns)
        R = (data - data.loc[0]) / stderrs
        left = R.apply(lambda col: np.percentile(col[1:], 84.135), axis=0)
        right = R.apply(lambda col: np.percentile(col[1:], 15.865), axis=0)
        left = data.loc[0] - stderrs.loc[0] * left
        right = data.loc[0] - stderrs.loc[0] * right
        CI.loc['left'] = left
        CI.loc['right'] = right
        return CI

    method = {'basic': basic, 'percentile': percentile, 't': student}
    method = method.pop(kind.lower(), basic)
    return method(*args)
