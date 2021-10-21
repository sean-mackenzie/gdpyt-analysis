# gdpyt-analysis: utils.fit
"""
Notes
"""

# imports
from scipy.optimize import curve_fit

# scripts


def fit(x, y, fit_function=None):

    # fit the function
    if fit_function is None:
        popt, pcov = curve_fit(parabola, x, y)
        fit_function = parabola
    else:
        popt, pcov = curve_fit(fit_function, x, y)

    return popt, pcov, fit_function


def fit_dficts(dficts, fit_function=None, xparameter=None, yparameter='z'):

    popts = []
    pcovs = []
    for name, df in dficts.items():

        # throw out NaNs
        df = df.dropna(axis=0, subset=[yparameter])

        # get x- and y-arrays
        if xparameter is None:
            x = df.index
        else:
            x = df[xparameter]
        y = df[yparameter]

        # fit the function
        if fit_function is None:
            popt, pcov = curve_fit(parabola, x, y)
            fit_function = parabola
        else:
            popt, pcov = curve_fit(fit_function, x, y)

        popts.append(popt)
        pcovs.append(pcov)

    return popts, pcovs, fit_function


def parabola(x, a, b, c):
    return a * x ** 2 + b * x + c