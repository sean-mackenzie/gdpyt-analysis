# gdpyt-analysis: utils.functions
"""
Notes
"""

# imports
import numpy as np


# --- define beam equations
E = 6e6  # elastic modulus of SILPURAN
t = 20e-6
L = 2.5e-3


def uniformly_loaded_beam_ss(x, P):
    return P * x / (24 * E * L * t ** 3 / 12) * (L ** 3 - 2 * L * x ** 2 + x ** 3)


def uniformly_loaded_beam_c(x, P):
    return P * x ** 2 / (24 * E * L * t ** 3 / 12) * (L - x) ** 2

# rectangular uniformly loaded thin plate
class fRectangularUniformLoad:

    def __init__(self, plate_width, youngs_modulus=6e6, plate_thickness=20e-6):
        self.L = plate_width
        self.E = youngs_modulus
        self.t = plate_thickness

    def rectangular_uniformly_loaded_clamped_plate(self, x, P):
        return P * x ** 2 / (24 * self.E * self.L * self.t ** 3 / 12) * (self.L - x) ** 2

    def rectangular_uniformly_loaded_simply_supported_plate(self, x, P):
        return P * x / (24 * self.E * self.L * self.t ** 3 / 12) * (self.L ** 3 - 2 * self.L * x ** 2 + x ** 3)


# spherical uniformly loaded thin plate
class fSphericalUniformLoad:

    def __init__(self, r, h, poisson=0.5):
        self.r = r
        self.h = h
        self.poisson = poisson

    def spherical_uniformly_loaded_clamped_plate(self, P, E):
        return P * self.r**4 / (64 * E * self.h**3 / (12 * (1 - self.poisson**2)))


    def spherical_uniformly_loaded_simply_supported_plate(self, P, E):
        return P * self.r**2 / (64 * E * self.h**3 / (12 * (1 - self.poisson**2))) * \
               ((5 + self.poisson) / (1 + self.poisson) * self.r**2)


# particle defocusing dependent on optics
class stigmatic_peak_intensity_of_z:
    """
    The following optical parameters need to be initialized:
    inst.n0 = refractive index of the medium (e.g., air = 1.0)
    inst.NA = numerical aperture (e.g., LCPLFLN20XLCD, NA=0.45)
    inst.pd = particle diameter (e.g., 2.15 micron NR = 2.15e-6)
    inst.wavelength = wavelength (e.g., = 600e-9)
    
    """

    def __init__(self):
        pass

    def intensity_profile(self, ref_index_medium):
        """self.c1 = 2 * (ref_index_medium ** 2 / numerical_aperture ** 2 - 1) ** -0.5
        self.c2 = (particle_diameter ** 2 + 1.49 * wavelength ** 2 * (
                    ref_index_medium ** 2 / numerical_aperture ** 2 - 1)) ** 0.5"""
        pass


# -------------------------------------------------- GENERAL FUCNTIONS -------------------------------------------------


def translate(x, a):
    return x - a


def line(x, a, b):
    return a * x + b


def parabola(x, a, b, c):
    return a * x ** 2 + b * x + c


def quadratic(x, a, b, c):
    return a * x ** 2 + b * x + c


def quadratic_slide(x, a, b, c, d):
    return a * (x + b) ** 2 + c * x + d


def cubic(x, a, b, c, d):
    return a * x ** 3 + b * x**2 + c * x + d


def quartic(x, a, b, c, d, e):
    return a * x ** 4 + b * x**4 + c * x**2 + d * x + e


def exponential(x, a, b, tau):
    return a * b ** (x / tau)


def general_gaussian_diameter(z, m, c1, c2):
    return m * np.sqrt(c1 ** 2 * z ** 2 + c2 ** 2)


# -------------------------------------------- PLANES AND QUASI-2D SURFACES --------------------------------------------


def plane(x, y, params):
    a = params[0]
    b = params[1]
    c = params[2]
    z = a*x + b*y + c
    return z


def smooth_surface(data, a, b, c):
    """
    3D curved surface function
    """
    x = data[0]
    y = data[1]
    return a * (x**b) * (y**c)


# ------------------------------------- CIRCLES, SPHERES, and ELLIPSOIDS -----------------------------------------------


def calculate_radius_at_xy(x, y, xc, yc):
    """
    Calculate radius (in x-y plane) from circle center.

    :param x:
    :param y:
    :param xc:
    :param yc:
    :return:
    """
    return np.sqrt((xc - x)**2 + (yc - y)**2)


def calculate_spherical_angle(r, xyz):
    """
    Given a point (x, y, z) approx. on a sphere of radius (r), return the angle phi and theta of that point.

    :param r:
    :param xyz:
    :return:
    """
    x, y, z = xyz[0], xyz[1], xyz[2]

    if np.abs(z) > r:
        return np.nan, np.nan
    else:
        phi = np.arccos(z / r)

        if x < 0 and y < 0:
            theta_half = np.arccos(x / (r * np.sin(phi)))
            theta_diff = np.pi - theta_half
            theta = np.pi + theta_diff
        else:
            theta = np.arccos(x / (r * np.sin(phi)))

    return phi, theta


# ------------------------------------- STATISTICAL ANALYSIS FUNCTIONS -------------------------------------------------


def calculate_precision(arr):
    return np.std(arr - np.mean(arr))


def plane_error(params, points):
    result = 0
    for (x, y, z) in points:
        plane_z = plane(x, y, params)
        diff = abs(plane_z - z)
        result += diff**2
    return result


def coeff_of_determination(x, y, fit_func, popt):
    # y fit
    y_fit = fit_func(x, *popt)

    # sum of the squared residuals
    ss_res = np.sum((y - y_fit) ** 2)

    # total sum of squares
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    # r-squared
    r2 = 1 - (ss_res / ss_tot)

    return r2, ss_res


# ---------------------------------------------- HELPER FUNCTIONS ------------------------------------------------------


def cross(a, b):
    return [a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]]


def calculate_z_of_3d_plane(x, y, popt):
    """
    Calculate the z-coordinate of a point lying on a 3D plane.

    :param x:
    :param y:
    :param popt:
    :return:
    """

    a, b, c, d, normal = popt[0], popt[1], popt[2], popt[3], popt[4]

    z = (-normal[0] * x - normal[1] * y - d) * 1. / normal[2]

    return z


def calculate_angle_between_planes(a, b, s=400):
    """
    I'm not sure if this works correctly for all cases (2/16/22)

    :param a:
    :param b:
    :param s:
    :return:
    """

    dz_xx = b[0, 0, 2] - a[0, 0, 2]
    dz_xy = b[0, 1, 2] - a[0, 1, 2]
    dz_yx = b[1, 0, 2] - a[1, 0, 2]
    dz_yy = b[1, 1, 2] - a[1, 1, 2]

    thetax = np.arctan((dz_xy - dz_xx) / s) * 360 / (2 * np.pi)
    thetay = np.arctan((dz_yx - dz_xx) / s) * 360 / (2 * np.pi)

    return thetax, thetay


def get_amplitude_center_sigma(z, intensity):

    # get amplitude
    raw_amplitude = intensity.max() - intensity.min()

    # get center
    raw_c = z[np.argmax(intensity)]

    # get sigma
    raw_profile_diff = np.diff(intensity)
    diff_peaks = np.argpartition(np.abs(raw_profile_diff), -2)[-2:]
    diff_width = np.abs(z[diff_peaks[1]] - z[diff_peaks[0]])
    raw_sigma = diff_width / 2

    return raw_amplitude, raw_c, raw_sigma