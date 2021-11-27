# gdpyt-analysis: utils.fit
"""
Notes
"""

# imports
import numpy as np
from scipy.optimize import curve_fit, minimize
import functools

# scripts


def fit(x, y, fit_function=None):

    # fit the function
    if fit_function is None:
        popt, pcov = curve_fit(parabola, x, y)
        fit_function = parabola
    else:
        popt, pcov = curve_fit(fit_function, x, y)

    return popt, pcov, fit_function


def fit_3d(points, fit_function):

    if fit_function == 'plane':
        fun = functools.partial(error, points=points)
        params0 = np.array([0, 0, 0])
        res = minimize(fun, params0)

        a = res.x[0]
        b = res.x[1]
        c = res.x[2]

        point = np.array([0.0, 0.0, c])
        normal = np.array(cross([1, 0, a], [0, 1, b]))
        d = -point.dot(normal)

        minx = np.min(points[:, 0])
        miny = np.min(points[:, 1])
        maxx = np.max(points[:, 0])
        maxy = np.max(points[:, 1])

        xx, yy = np.meshgrid([50, 450], [50, 450])
        z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

        return xx, yy, z


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


def line(x, a, b):
    return a * x + b


def plane(x, y, params):
    a = params[0]
    b = params[1]
    c = params[2]
    z = a*x + b*y + c
    return z


def error(params, points):
    result = 0
    for (x, y, z) in points:
        plane_z = plane(x, y, params)
        diff = abs(plane_z - z)
        result += diff**2
    return result


def cross(a, b):
    return [a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]]