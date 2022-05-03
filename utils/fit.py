# gdpyt-analysis: utils.fit
"""
Notes
"""

# imports
import math
import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import SmoothBivariateSpline
import functools

from utils import functions


# scripts


def gauss_1d_function(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def fit(x, y, fit_function=None, bounds=None):
    # fit the function
    if fit_function is None:
        popt, pcov = curve_fit(functions.parabola, x, y)
        fit_function = functions.parabola
    else:
        if bounds is not None:
            popt, pcov = curve_fit(fit_function, x, y, bounds=bounds)
        else:
            popt, pcov = curve_fit(fit_function, x, y)

    return popt, pcov, fit_function


def fit_3d(points, fit_function):
    return fit_3d_plane(points)


def fit_3d_plane(points):
    fun = functools.partial(functions.plane_error, points=points)
    params0 = np.array([0, 0, 0])
    res = minimize(fun, params0)

    a = res.x[0]
    b = res.x[1]
    c = res.x[2]

    point = np.array([0.0, 0.0, c])
    normal = np.array(functions.cross([1, 0, a], [0, 1, b]))
    d = -point.dot(normal)

    popt = [a, b, c, d, normal]

    minx = np.min(points[:, 0])
    miny = np.min(points[:, 1])
    maxx = np.max(points[:, 0])
    maxy = np.max(points[:, 1])

    xx, yy = np.meshgrid([minx, maxx], [miny, maxy])
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

    return xx, yy, z, popt


def fit_3d_spline(x, y, z, kx=1, ky=2):
    w = np.ones_like(x)
    bispl = SmoothBivariateSpline(x, y, z, w=w, kx=kx, ky=ky)
    rmse = np.sqrt(bispl.get_residual() / len(x))
    return bispl, rmse


def fit_3d_sphere(X, Y, Z):
    """
    Fit a sphere to data points (X, Y, Z) and return the sphere radius and center of best fit.

    Reference: Charles Jekel (2016); https://jekel.me/2015/Least-Squares-Sphere-Fit/

    :param X:
    :param Y:
    :param Z:
    :return:
    """
    #   Assemble the A matrix
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    A = np.zeros((len(X), 4))
    A[:, 0] = X * 2
    A[:, 1] = Y * 2
    A[:, 2] = Z * 2
    A[:, 3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(X), 1))
    f[:, 0] = (X * X) + (Y * Y) + (Z * Z)
    C, residules, rank, singval = np.linalg.lstsq(A, f)
    xc, yc, zc = C[0], C[1], C[2]

    #   solve for the radius
    t = (xc * xc) + (yc * yc) + (zc * zc) + C[3]
    radius = math.sqrt(t)

    return radius, xc, yc, zc


def fit_3d_sphere_from_center(spX, spY, spZ, xc, yc):
    """
    Fit a sphere to data points (spX, spY, spZ) given the sphere center in x-y coordinates.

    :param spX:
    :param spY:
    :param spZ:
    :param xc:
    :param yc:
    :return:
    """
    #   Assemble the A matrix
    spX = np.array(spX)
    spY = np.array(spY)
    spZ = np.array(spZ)

    A = np.zeros((len(spX), 2))
    A[:, 0] = spZ * 2
    A[:, 1] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX), 1))
    f[:, 0] = (spX * spX) + (spY * spY) + (spZ * spZ) - (2 * spX * xc) - (2 * spY * yc)

    # least squares fit
    C, residules, rank, singval = np.linalg.lstsq(A, f)
    zc = C[0]

    #   solve for the radius
    t = (xc ** 2) + (yc ** 2) + (zc ** 2) + C[1]
    radius = math.sqrt(t)

    return radius, C[0]


def fit_ellipsoid_from_center(X, Y, Z, xc, yc, zc, r):
    """
    Fit a 3D ellipsoid given the x, y, z center coordinates, x-radius, and y-radius.

    Somewhat helpful reference: https://jekel.me/2020/Least-Squares-Ellipsoid-Fit/
    :param X:
    :param Y:
    :param Z:
    :param xc:
    :param yc:
    :param zc:
    :param r:
    :return:
    """
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    f = np.zeros((len(X), 1))
    f[:, 0] = -1 * ((Z * Z) - (2 * zc * Z) + (zc * zc))

    A = np.zeros((len(X), 1))
    A[:, 0] = ((X * X) - (2 * xc * X) + (xc * xc) + (Y * Y) - (2 * yc * Y) + (yc * yc)) / (r * r) - 1

    # least squares fit
    C, residules, rank, singval = np.linalg.lstsq(A, f)

    # solve for radius in z-dir.
    r_z = math.sqrt(C[0])

    return r_z


def fit_ellipsoid_non_linear(X, Y, Z):
    """
    Non-linear regression + optimization to find ellipsoid parameters.
    Reference: https://jekel.me/2021/A-better-way-to-fit-Ellipsoids/

    :param X:
    :param Y:
    :param Z:
    :return:
    """
    x, y, z = X, Y, Z
    pass


def fit_smooth_surface(df, z_param='z'):
    """
    Uses the 'smooth_surface' fit function on a dataframe.

    :param df:
    :return:
    """

    # convert data into proper format
    x_data = df.x.to_numpy()
    y_data = df.y.to_numpy()
    z_data = df[z_param].to_numpy()

    data = [x_data, y_data]

    # get fit parameters from scipy curve fit
    fittedParameters, covariance = curve_fit(functions.smooth_surface, data, z_data)

    # --- calculate prediction errors
    rmse, r_squared = calculate_fit_error(fit_results=None,
                                          data_fit_to=z_data,
                                          fit_func=functions.smooth_surface,
                                          fit_params=fittedParameters,
                                          data_fit_on=data,
                                          )

    # deprecated prediction errors
    """modelPredictions = functions.smooth_surface(data, *fittedParameters)
    absError = modelPredictions - z_data
    SE = np.square(absError)  # squared errors
    MSE = np.mean(SE)  # mean squared errors
    RMSE = np.sqrt(MSE)  # Root Mean Squared Error, RMSE
    Rsquared = 1.0 - (np.var(absError) / np.var(z_data))"""

    return rmse, r_squared, fittedParameters


# ----------------------------------------------- HELPER FUNCTIONS -----------------------------------------------------

def calculate_fit_error(fit_results, data_fit_to, fit_func=None, fit_params=None, data_fit_on=None):
    """
    Two options for calculating fit error:
        1. fit_func + fit_params: the fit results are calculated.
        2. fit_results: the fit results are known for each data point.

    :param fit_func: the function used to calculate the fit.
    :param fit_params: generally, popt.
    :param fit_results: the outputs at each input data point ('data_fit_on')
    :param data_fit_on: the input data that was inputted to fit_func to generate the fit.
    :param data_fit_to: the output data that fit_func was fit to.
    :return:
    """

    # --- calculate prediction errors
    if fit_results is None:
        fit_results = fit_func(data_fit_on, *fit_params)

    abs_error = fit_results - data_fit_to
    se = np.square(abs_error)  # squared errors
    mse = np.mean(se)  # mean squared errors
    rmse = np.sqrt(mse)  # Root Mean Squared Error, RMSE
    r_squared = 1.0 - (np.var(abs_error) / np.var(data_fit_to))

    return rmse, r_squared


# ----------------------------------------------- DEPRECATED FUNCTIONS -------------------------------------------------


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
            popt, pcov = curve_fit(functions.parabola, x, y)
            fit_function = functions.parabola
        else:
            popt, pcov = curve_fit(fit_function, x, y)

        popts.append(popt)
        pcovs.append(pcov)

    return popts, pcovs, fit_function