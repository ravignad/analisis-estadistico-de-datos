# This module provides a collection of utility functions for statistical data analysis
# including linear least squares fitting, covariance ellipse calculation, cost functions, and more.
# It also includes functions for analyzing parameter estimators and statistical properties.

# Standard Library Imports
import math
from pathlib import PurePath
from typing import Tuple

# Third-Party Imports
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from scipy.stats import chi2
from scipy.stats import binned_statistic


def get_ellipse(center: np.ndarray, cova: np.ndarray, nsigma: int = 1, npoints: int = 1000) -> np.ndarray:
    """
    Return the coordinates of a covariance ellipse.

    Args:
        center (np.ndarray): The center of the ellipse.
        cova (np.ndarray): The covariance matrix.
        nsigma (int, optional): The number of standard deviations for the ellipse. Defaults to 1.
        npoints (int, optional): The number of points to generate on the ellipse. Defaults to 1000.

    Returns:
        np.ndarray: The coordinates of the ellipse.
    """
    cholesky_l = np.linalg.cholesky(cova)
    t = np.linspace(0, 2 * np.pi, npoints)
    circle = np.column_stack([np.cos(t), np.sin(t)])
    ellipse = nsigma * circle @ cholesky_l.T + center
    return ellipse.T


def linear_least_squares(model_matrix: np.ndarray, y: np.ndarray, ysigma: np.ndarray) -> dict:
    """
    Fit data with a linear least squares method.

    Args:
        model_matrix (np.ndarray): Model/design/X matrix.
        y (np.ndarray): Independent variable.
        ysigma (np.ndarray): Y errors.

    Returns:
        dict: Dictionary containing parameter estimators, errors, covariance matrix, correlation matrix,
        chi-squared minimum, degrees of freedom, and p-value.
    """

    # Parameter estimators
    cova_y = np.diag(ysigma * ysigma)
    cova_par = inv(model_matrix.T @ inv(cova_y) @ model_matrix)
    matrix_b = cova_par @ model_matrix.T @ inv(cova_y)
    theta_est = matrix_b @ y

    # Future implementation with Einstein summation (to be used for large datasets)
    # inv_var_y = ysigma**(-2)
    # inv_cova_par2 = np.einsum('ji,j,jl', model_matrix, inv_var_y, model_matrix)
    # cova_par2 = inv(inv_cova_par2)
    # matrix_b2 = np.einsum('ij,kj,k -> ik', cova_par2, model_matrix, inv_var_y)
    # theta_est2 = np.einsum('ij,j', matrix_b2, y)

    # Parameter errors
    errors = np.sqrt(np.diagonal(cova_par))
    corr = cova_par / np.tensordot(errors, errors, axes=0)

    # Goodness of fit
    residuals = y - model_matrix @ theta_est
    chi2_min = residuals.T @ inv(cova_y) @ residuals
    ndof = len(y) - len(theta_est)
    pvalue = chi2.sf(chi2_min, ndof)

    return {
        'est': theta_est,
        'errors': errors,
        'cova': cova_par,
        'corr': corr,
        'chi2_min': chi2_min,
        'ndof': ndof,
        'pvalue': pvalue
    }


def cost_poisson(y: np.ndarray, mu: np.ndarray) -> float:
    """
    Cost function of a Poisson variable.

    Args:
        y (np.ndarray): Measured values of the dependent variable.
        mu (np.ndarray): Model values at each measured data point.

    Returns:
        float: Fit cost.
    """
    cost_array = 2 * (mu - y) - 2 * y * np.log(mu / y)
    return cost_array.sum()


def fit_errors(grad: np.ndarray, cova: np.ndarray) -> np.ndarray:
    """
    Calculate the fit errors by propagating the parameter errors.

    Args:
        grad (np.ndarray): Gradient of the fit model with respect to the parameters.
        cova (np.ndarray): Covariance matrix of the fit parameters.

    Returns:
        np.ndarray: Fit errors.
    """
    var_mu_est = np.einsum("ki,ij,kj->k", grad, cova, grad)
    sigma_mu_est = np.sqrt(var_mu_est)
    return sigma_mu_est


def get_bias(estimators: np.ndarray, parameter: float) -> Tuple[float, float]:
    """
    Estimate the bias of a parameter estimator.

    Args:
        estimators (np.ndarray): Array of parameter estimators.
        parameter (float): True parameter value.

    Returns:
        tuple: Bias and standard error of the estimator.
    """
    mean = np.mean(estimators)
    bias = mean - parameter
    sigma = np.std(estimators, ddof=1)
    sigma_mean = sigma / math.sqrt(len(estimators))
    return bias, sigma_mean


def get_coverage(estimators: np.ndarray, errors: np.ndarray, parameter: float) -> Tuple[float, float]:
    """
    Estimate the coverage of the confidence intervals.

    Args:
        estimators (np.ndarray): Array of parameter estimators.
        errors (np.ndarray): Array of parameter errors.
        parameter (float): True parameter value.

    Returns:
        tuple: Coverage and its standard error.
    """
    theta_min = np.array(estimators) - np.array(errors)
    theta_max = np.array(estimators) + np.array(errors)
    hits = ((parameter - theta_min) * (parameter - theta_max) < 0).sum()
    ndata = len(estimators)
    coverage = hits / ndata
    coverage_error = math.sqrt(coverage * (1 - coverage) / ndata)
    return coverage, coverage_error


def get_pvalue(chi2_sim: list, chi2_obs: float) -> Tuple[float, float]:
    """
    Calculate the p-value of a chi-squared test.

    Args:
        chi2_sim (list): List of simulated chi-squared values.
        chi2_obs (float): Observed chi-squared value.

    Returns:
        tuple: P-value and its standard error.
    """
    ntail = np.sum(np.array(chi2_sim) > chi2_obs)
    ndata = len(chi2_sim)
    pvalue = ntail / ndata
    pvalue_error = math.sqrt(pvalue * (1 - pvalue) / ndata)
    return pvalue, pvalue_error


def savefigs(basename: str, formats: tuple = ('.eps', '.pdf', '.png', '.svg'), folder: str = '') -> None:
    """
    Save a figure to multiple formats and print their names.

    Args:
        basename (str): Base filename for the saved figures.
        formats (tuple, optional): Tuple of file formats to save. Defaults to ('.eps', '.pdf', '.png', '.svg').
        folder (str, optional): Folder where the figures will be saved. Defaults to ''.
    """
    fig = plt.gcf()
    for fig_format in formats:
        figure_name = PurePath(folder, basename + fig_format)
        print(f'Saving figure to {figure_name}')
        fig.savefig(figure_name)


def profile_histogram(x: np.ndarray, y: np.ndarray, bins: int, histo_range: tuple = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a profile histogram.

    Args:
        x (np.ndarray): The x-values.
        y (np.ndarray): The y-values.
        bins (int): The number of bins.
        histo_range (tuple, optional): Range for the histogram. Defaults to None.

    Returns:
        tuple: Counts, means, standard deviations, and bin edges.
    """

    counts, bin_edges, __ = binned_statistic(x, y, statistic='count', bins=bins, range=histo_range)
    means, __, __ = binned_statistic(x, y, statistic='mean', bins=bins, range=histo_range)
    means2, __, __ = binned_statistic(x, y*y, statistic='mean', bins=bins, range=histo_range)

    # Standard deviations with Bessel correction
    variances = counts * (means2 - means**2) / (counts-1)
    standard_deviations = np.sqrt(variances)
    
    return counts, means, standard_deviations, bin_edges


def array_rms(array):
    """
    Calculate the root of the mean of the squares of an array

    Args:
        array (np.array): input array

    Returns:
         square root of the mean of the array elements
    """
    return math.sqrt(np.mean(array ** 2))
