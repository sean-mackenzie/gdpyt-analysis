# gdpyt-analysis: utils.functions
"""
Notes
"""

# imports
import numpy as np
from scipy import special
from scipy.optimize import curve_fit
from utils import fit


# ---

# --------------------------------------------- PLATE THEORY  ----------------------------------------------------------


# Linear Plate Theory: rectangular uniformly loaded thin plate

class fRectangularUniformLoad:

    def __init__(self, plate_width, youngs_modulus=5e6, plate_thickness=20e-6):
        self.L = plate_width
        self.E = youngs_modulus
        self.t = plate_thickness

    def rectangular_uniformly_loaded_clamped_plate(self, x, P):
        return P * x ** 2 / (24 * self.E * self.L * self.t ** 3 / 12) * (self.L - x) ** 2

    def rectangular_uniformly_loaded_simply_supported_plate(self, x, P):
        return P * x / (24 * self.E * self.L * self.t ** 3 / 12) * (self.L ** 3 - 2 * self.L * x ** 2 + x ** 3)


# ---


# Linear Plate Theory: spherical uniformly loaded thin plate

class fSphericalUniformLoad:

    def __init__(self, r, h, youngs_modulus=None, poisson=0.5):
        self.r = r
        self.h = h
        self.E = youngs_modulus
        self.poisson = poisson

    def spherical_uniformly_loaded_clamped_plate_r_p(self, r, P):
        return P * (self.r ** 2 - r ** 2) ** 2 / (64 * self.E * self.h ** 3 / (12 * (1 - self.poisson ** 2)))

    def spherical_uniformly_loaded_simply_supported_plate_r_p(self, r, P):
        return P * (self.r ** 2 - r ** 2) / (64 * self.E * self.h ** 3 / (12 * (1 - self.poisson ** 2))) * \
               (((5 + self.poisson) / (1 + self.poisson)) * self.r ** 2 - r ** 2)

    def spherical_uniformly_loaded_clamped_plate_p_e(self, P, E):
        return P * self.r ** 4 / (64 * E * self.h ** 3 / (12 * (1 - self.poisson ** 2)))

    def spherical_uniformly_loaded_simply_supported_plate_p_e(self, P, E):
        return P * self.r ** 2 / (64 * E * self.h ** 3 / (12 * (1 - self.poisson ** 2))) * \
               ((5 + self.poisson) / (1 + self.poisson) * self.r ** 2)

    # plotting

    def plot_dimensional_z_by_r_p_k(self, d_r, d_p0, d_n0):
        """ d_z = .plot_dimensional_z_by_r_p_k(d_r, d_p0, d_n0) """
        # calculate non-dimensional deflection
        d_z = self.spherical_uniformly_loaded_simply_supported_plate_r_p(d_r, d_p0)

        return d_z


# ---


# Linear and Nonlinear Plate Theory: circular uniformly loaded thin plate or membrane

class fNonDimensionalNonlinearSphericalUniformLoad:

    def __init__(self, r, h, youngs_modulus=None, poisson=0.5):
        self.r = r
        self.h = h
        self.E = youngs_modulus
        self.poisson = poisson
        self.D = youngs_modulus * h ** 3 / (12 * (1 - poisson))
        self.C1 = 12 * (1 - poisson ** 2) / 4

    def non_dimensionalize_p_k(self, d_p0, d_n0):
        """ nd_P, nd_k = .non_dimensionalize_p_k(d_p0, d_n0) """
        nd_P = d_p0 * self.r ** 4 / (self.E * self.h ** 4)
        nd_k = np.sqrt(d_n0 * self.r ** 2 / self.D)
        return nd_P, nd_k

    def nd_nonlinear_clamped_plate_p_k(self, nd_r, nd_P, nd_k):
        return self.C1 * nd_P * (
                (2 * (special.iv(0, nd_k * nd_r) - special.iv(0, nd_k))) / (nd_k ** 3 * special.iv(1, nd_k)) +
                ((1 - nd_r ** 2) / nd_k ** 2)
        )

    def nd_nonlinear_clamped_plate_p_k_lva(self, nd_r, nd_P, nd_k):
        return self.C1 * nd_P / nd_k ** 2 * \
               (1 - nd_r ** 2 - 2 / nd_k * (1 - np.exp(-nd_k * (1 - nd_r)) / np.sqrt(nd_r)))

    # models

    def nd_linear_z_plate(self, nd_r, nd_P):
        return self.C1 * .0625 * nd_P * (1 - nd_r ** 2) ** 2

    def nd_linear_z_membrane(self, nd_r, nd_P, nd_k):
        return self.C1 * nd_P * (1 - nd_r ** 2) / nd_k ** 2

    def nd_nonlinear_theta(self, nd_r, nd_P, nd_k):
        return self.C1 * 2 * nd_P * (special.iv(1, nd_k * nd_r) / (nd_k ** 2 * special.iv(1, nd_k)) - nd_r / nd_k ** 2)

    def nd_nonlinear_theta_lva(self, nd_r, nd_P, nd_k):
        return -2 * self.C1 * nd_P / nd_k ** 2 * (nd_r - np.exp(-nd_k * (1 - nd_r)) / np.sqrt(nd_r))

    def nd_nonlinear_theta_plate(self, nd_r, nd_P):
        nd_theta_plate = self.C1 * -0.25 * nd_P * nd_r * (1 - nd_r ** 2)
        return nd_theta_plate / np.max(np.abs(nd_theta_plate)) * 1.5

    def nd_nonlinear_theta_membrane(self, nd_r, nd_P):
        nd_k = 100
        nd_theta_memb = self.C1 * -2 * nd_P * nd_r / nd_k ** 2
        return nd_theta_memb / np.max(np.abs(nd_theta_memb)) * 2

    def nd_nonlinear_curvature_lva(self, nd_r, nd_P, nd_k):
        return self.C1 * 2 * nd_P / nd_k ** 2 * (-1 + (nd_k * nd_r - 1) * np.exp(-nd_k * (1 - nd_r)) / nd_r ** (3 / 2))

    # calculate other quantities

    def calculate_radial_loads(self, nd_k):
        nd_Sr = self.calculate_nd_Sr_radial_load(nd_k)
        Nr = self.calculate_Nr_radial_load(nd_Sr)
        sigma_r = self.calculate_radial_stress(Nr)
        eta_r = self.calculate_radial_strain(sigma_r)
        return {'nd_Sr': nd_Sr, 'Nr': Nr, 'sigma_r': sigma_r, 'eta_r': eta_r}

    def calculate_nd_Sr_radial_load(self, nd_k):
        return nd_k ** 2 / self.C1 * 4

    def calculate_Nr_radial_load(self, nd_Sr):
        return nd_Sr * self.E * self.h ** 3 / self.r ** 2

    def calculate_radial_stress(self, Nr):
        return Nr / self.h

    def calculate_radial_strain(self, sigma_r):
        return sigma_r * (1 - self.poisson ** 2) / self.E

    # fit

    def fit_nd_nonlinear(self, model, nd_r, nd_z, nd_r_eval=None, nd_z_eval=None,
                         guess=(-50, 5), bounds=([-1e9, 0], [1e9, 1e9])):
        """ fit_d_r, fit_d_z, d_p0, d_n0, rmse, r_squared = .fit_nd_nonlinear(nd_r, nd_z) """
        # fit
        if model == 'lva':
            func = self.nd_nonlinear_clamped_plate_p_k_lva
        else:
            func = self.nd_nonlinear_clamped_plate_p_k

        popt, pcov = curve_fit(func, nd_r, nd_z, p0=guess, bounds=bounds, xtol=1.49012e-07, maxfev=1000)
        nd_P, nd_k = popt[0], popt[1]

        # calculate the fit error
        if nd_r_eval is not None:
            rmse, r_squared = fit.calculate_fit_error(
                fit_results=func(nd_r_eval, *popt),
                data_fit_to=nd_z_eval,
            )
        else:
            rmse, r_squared = fit.calculate_fit_error(
                fit_results=func(nd_r, *popt),
                data_fit_to=nd_z,
            )

        # resample
        fit_nd_r = np.linspace(0, self.r, 250) / self.r
        fit_nd_z = func(fit_nd_r, *popt)

        # dimensionalize
        rmse = rmse * self.h
        fit_d_r = fit_nd_r * self.r
        fit_d_z = fit_nd_z * self.h
        d_p0 = nd_P * self.E * self.h ** 4 / self.r ** 4
        d_n0 = nd_k ** 2 * self.D / self.r ** 2

        return fit_d_r, fit_d_z, d_p0, d_n0, rmse, r_squared

    # plotting

    def plot_dimensional_z_by_r_p_k(self, d_r, d_p0, d_n0):
        """ d_z = .plot_dimensional_z_by_r_p_k(d_r, d_p0, d_n0) """
        # non-dimensionalize variables
        nd_r = d_r / self.r
        nd_P, nd_k = self.non_dimensionalize_p_k(d_p0, d_n0)

        # calculate non-dimensional deflection
        nd_z = self.nd_nonlinear_clamped_plate_p_k(nd_r, nd_P, nd_k)

        # dimensionalize deflection
        d_z = nd_z * self.h

        return d_z

# ---


# -------------------------------------------- OPTICS ------------------------------------------------------------------


# ---


def depth_of_field(M, NA, wavelength, n0, pixel_size):
    """ First term: Diffraction-limited DoF (wave optics), second term: Cirlce-of-confusion-limited DoF (geometric) """
    return wavelength * n0 / NA ** 2 + n0 * pixel_size / (M * NA)


def particle_image_diameter_by_z(z, magnification, numerical_aperture, p_d, wavelength, n0=1.0):
    """

    :param z: axial position
    :param mag_eff: effective magnification
    :param na_eff: effective numerical aperture
    :param p_d: particle diameter (the physical particle)
    :param wavelength: emission wavelength
    :param n_0: refraction index of the immersion medium (air: n_0 = 1.0)
    :return:
    """
    return magnification * (p_d ** 2 +
                            1.49 * wavelength ** 2 * (n0 ** 2 / numerical_aperture ** 2 - 1) +
                            4 * z ** 2 * (n0 ** 2 / numerical_aperture ** 2 - 1) ** -1
                            ) ** 0.5


def particle_image_diameter_term_by_z(term, z, magnification, numerical_aperture, p_d, wavelength, n0=1.0):
    dia_p = p_d * magnification
    dia_s = 1.22 * magnification * wavelength * (n0 ** 2 / numerical_aperture ** 2 - 1) ** 0.5
    dia_f = magnification * (4 * z ** 2 * (n0 ** 2 / numerical_aperture ** 2 - 1) ** -1) ** 0.5

    if term == 'geometric':
        return dia_p * np.ones_like(z)
    elif term == 'diffraction':
        return dia_s * np.ones_like(z)
    elif term == 'defocused':
        return dia_f
    else:
        return dia_p + dia_s + dia_f


# ---

# Particle image optics given optical configuration

class fParticleImageOpticalTheory:

    def __init__(self, magnification, numerical_aperture, particle_diameter, wavelength, pixel_size,
                 bkg_mean, bkg_noise, n0=1.0):
        """
        Notes:
            20X - LCPlanFL N 20X LCD        [LCPLFLN20xLCD]
                magnification:              20
                numerical_aperture:         0.45
                field_number:               26.5
                working distance:           7.4 - 8.3 mm
                transmittance:              90% @ 425 - 670 nm
                correction collar:          0 - 1.2 mm
                objective lens diameter:    15 mm
                microns per pixel:          0.8     (1.6 w/ 0.5X demagnifier)

        :param magnification:
        :param numerical_aperture:
        :param particle_diameter:
        :param wavelength:
        :param pixel_size:
        :param n0:
        """
        self.magnification = magnification
        self.numerical_aperture = numerical_aperture
        self.particle_diameter = particle_diameter
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.n0 = n0

        # additional parameters
        self.depth_of_field = None
        self.c1 = None
        self.c2 = None
        self.Rayleigh_criterion = None

    def particle_image_diameter_by_z(self, z):
        """
        This is the actual equation describing particle image diameter. The other equation uses c1 and c2.
            > 'ideally' these would be identical

        :param z:
        :return:
        """
        return self.magnification * (self.particle_diameter ** 2 +
                                     1.49 * self.wavelength ** 2 * (self.n0 ** 2 / self.numerical_aperture ** 2 - 1) +
                                     4 * z ** 2 * (self.n0 ** 2 / self.numerical_aperture ** 2 - 1) ** -1
                                     ) ** 0.5

    def fit_effective_numerical_aperture(self, z, na_eff):
        return self.magnification * (self.particle_diameter ** 2 +
                                     1.49 * self.wavelength ** 2 * (self.n0 ** 2 / na_eff ** 2 - 1) +
                                     4 * z ** 2 * (self.n0 ** 2 / na_eff ** 2 - 1) ** -1
                                     ) ** 0.5

    def calculate_depth_dependent_stigmatic_diameter(self, z):
        """
         dia_theory = fPI.calculate_depth_dependent_stigmatic_diameter(z) * 1e6 / mag_eff

        NOTE: the result needs to be multiplied by 1e6 and divided by the magnification (see example above)

        particle image diameter with distance from focal plane (stigmatic system)
            Ref: Rossi & Kahler 2014, DOI 10.1007 / s00348-014-1809-2)
        """
        if self.c1 is None:
            self.calculate_theoretical_c1_c2()

        return self.magnification * (self.c1 ** 2 * (z * 1e-6) ** 2 + self.c2 ** 2) ** 0.5

    def calculate_depth_dependent_stigmatic_peak_intensity(self, z):
        """
        int_theory = fPI.calculate_depth_dependent_stigmatic_peak_intensity(z) * np.max(int_exp_fit)
            > where 'int_exp_fit' is the experimental maximum intensity.

        ref: Rossi & Kahler 2014, DOI 10.1007/s00348-014-1809-2
        """
        if self.c1 is None:
            self.calculate_theoretical_c1_c2()

        return self.c2 ** 2 / ((self.c1 ** 2 * (z * 1e-6) ** 2 + self.c2 ** 2) **
                               0.5 * (self.c1 ** 2 * (z * 1e-6) ** 2 +
                                      self.c2 ** 2) ** 0.5)

    def depth_of_field(self):
        """
        First term: Diffraction-limited DoF (wave optics),
        second term: Cirlce-of-confusion-limited DoF (geometric)
        """
        self.depth_of_field = self.wavelength * self.n0 / self.numerical_aperture ** 2 + \
                              self.n0 * self.pixel_size / (self.magnification * self.numerical_aperture)

        return self.depth_of_field

    def calculate_theoretical_c1_c2(self):
        """
        constants for stigmatic/astigmatic imaging systems (ref: Rossi & Kahler 2014, DOI 10.1007/s00348-014-1809-2)
        """
        self.c1 = 2 * (self.n0 ** 2 / self.numerical_aperture ** 2 - 1) ** -0.5
        self.c2 = (
                          (self.particle_diameter ** 2 + 1.49 * self.wavelength ** 2 * (
                                  self.n0 ** 2 / self.numerical_aperture ** 2 - 1)) ** 0.5
                  )

    def calculate_diameter_from_sigma(self, A, sigma):
        beta_squared = 3.67

        x_arr = np.linspace(0, sigma * 3, 1000)
        x_intensity = gauss_1d(x=x_arr, a=A, x0=0, sigma=sigma)
        x_intensity_raw = x_intensity - np.exp(-beta_squared) * A
        x_intensity_rel = np.abs(x_intensity_raw)
        dia = x_arr[np.argmin(x_intensity_rel)]

        return dia

    def set_numerical_aperture(self, effective_numerical_aperture):
        self.numerical_aperture = effective_numerical_aperture

    def set_c1_c2(self, c1, c2):
        self.c1 = c1
        self.c2 = c2

    def calculate_rayleigh_criterion(self):
        """
        The Rayleigh Criterion is the maximum lateral resolution (distance) that two point sources can be discerned.
        It's defined as the distance between the distance where the principal diffraction maximum (central spot of the
        Airy disk) from one point source overlaps with the first minimum (dark region surrounding the central spot) from
        the Airy disk of the other point source.
        """
        self.Rayleigh_criterion = 0.61 * self.wavelength / self.numerical_aperture

        return self.Rayleigh_criterion

    def lower_bound_position_uncertainty(self, gauss_sigma, b, N):
        """
        Reference: Centroid precision and orientation precision of planar localization microscopy

        :param gauss_sigma: the standard deviation of the Gaussian approximation of the microscope PSF
        :param b:  b --> b ** 2: the expected number of background photons per pixel
        :param N: the total number of signal photons
        :param a: the pixel pitch of the imaging sensor
        :return:
        """
        a = self.pixel_size

        L_p = np.sqrt(
            (16 * (gauss_sigma ** 2 + a ** 2 / 12) / (9 * N)) +
            (8 * np.pi * b ** 2 * (gauss_sigma ** 2 + a ** 2 / 12) ** 2 / (a ** 2 * N ** 2))
        )

        return L_p


class stigmatic_peak_intensity_of_z:
    """ particle defocusing dependent on optics:

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
    return a * (x + b) ** 2 + c * (x + b) + d


def cubic(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d


def cubic_slide(x, a, b, c, d, e):
    return a * (x + e) ** 3 + b * (x + e) ** 2 + c * (x + e) + d


def quartic(x, a, b, c, d, e):
    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e


def quartic_slide(x, a, b, c, d, e, f):
    return a * (x + f) ** 4 + b * (x + f) ** 3 + c * (x + f) ** 2 + d * (x + f) + e


def exponential(x, a, b, tau):
    return a * b ** (x / tau)


def gauss_1d(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def general_gaussian_diameter(z, m, c1, c2):
    return m * np.sqrt(c1 ** 2 * z ** 2 + c2 ** 2)


# --- TRIGONOMETRIC

def sine(x, A, f, phi):
    """ Requires a good guess to be accurate """
    return A * np.sin(2 * np.pi * f * x + phi)


def fit_sin(tt, yy, guess=None, bounds=None):
    """
    dict_res = functions.fit_sin(tt, yy)
    A = dict_res['amp']
    f = dict_res['freq']
    period = dict_res['period']
    fit_func = dict_res['fitfunc']
    sine_func = dict_res['sinfunc']
    popt = dict_res['popt']

    :param tt:
    :param yy:
    :return:
    """
    tt = np.array(tt)
    yy = np.array(yy)

    if guess is None:
        ff = np.fft.fftfreq(len(tt), (tt[1] - tt[0]))  # assume uniform spacing
        Fyy = abs(np.fft.fft(yy))
        guess_freq = abs(ff[np.argmax(Fyy[1:]) + 1])  # excluding the zero frequency "peak", which is related to offset
        guess_amp = np.std(yy) * 2. ** 0.5
        guess_offset = np.mean(yy)
        guess = np.array([guess_amp, 2. * np.pi * guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):
        return A * np.sin(w * t + p) + c

    if bounds is not None:
        popt, pcov = curve_fit(sinfunc, tt, yy, p0=guess, bounds=bounds)
    else:
        popt, pcov = curve_fit(sinfunc, tt, yy, p0=guess)

    # results
    A, w, p, c = popt
    f = w / (2. * np.pi)
    fitfunc = lambda t: A * np.sin(w * t + p) + c
    sinfunc = sinfunc

    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1. / f,
            "fitfunc": fitfunc, "sinfunc": sinfunc, "popt": popt,
            "maxcov": np.max(pcov), "rawres": (guess, popt, pcov)}


# -------------------------------------------- PLANES AND QUASI-2D SURFACES --------------------------------------------


def plane(x, y, params):
    a = params[0]
    b = params[1]
    c = params[2]
    z = a * x + b * y + c
    return z


def smooth_surface(data, a, b, c):
    """
    3D curved surface function
    """
    x = data[0]
    y = data[1]
    return a * (x ** b) * (y ** c)


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
    return np.sqrt((xc - x) ** 2 + (yc - y) ** 2)


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


def calculate_residuals(fit_results, data_fit_to):
    residuals = fit_results - data_fit_to
    return residuals


def calculate_precision(arr):
    return np.std(arr - np.mean(arr))


def calculate_coefficient_of_variation(arr):
    return np.std(arr) / np.mean(arr)


def calculate_t_test(a, b):
    """
    :param a: array
    :param b: array

    Research hypothesis: the average of (a) is greater than the average of (b)
    Null hypothesis: there is no difference between the averages of (a) and (b)
    "One-tailed" - we expect (a) to be greater than (b)
    "Two-tailed" - we expect a difference but don't know if (a) or (b) is greater

    Reference: https://home.csulb.edu/~msaintg/ppa696/696stsig.htm
    """

    # t-test
    t = (np.mean(b) - np.mean(a)) / np.sqrt(np.var(a) / (len(a) - 1) + np.var(b) / (len(b) - 1))

    # degrees of freedom
    dof = len(a) + len(b) - 2

    return t, dof


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
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def plane_error(params, points):
    result = 0
    for (x, y, z) in points:
        plane_z = plane(x, y, params)
        diff = abs(plane_z - z)
        result += diff ** 2
    return result


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


def df_weighted_average(df, column_of_weights):
    """
    Return a one-row dataframe of the average of all columns using 'column_of_weights' as weights for average.
    :param df:
    :param column_of_weights:
    :return:
    """
    columns = df.columns
    series_bin = df.groupby('tid').apply(lambda x: np.average(x, axis=0, weights=x.f_num_pids)).to_numpy()
    series_data = []
    for sb in series_bin:
        series_data.append(sb)
    series_data_arr = np.array(series_data)
    df_bin = pd.DataFrame(series_data_arr, columns=columns)