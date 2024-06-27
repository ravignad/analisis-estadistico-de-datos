# Evaluate the performance of a fit

import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from scipy.stats import poisson
from scipy.stats import chi2

import likefit
import danatools

# Switch off interactive plotting
plt.ioff()

# Number of simulations
SIMULATIONS = 1000


def main():

    # Fitted parameters
    S0_true = 19.1
    beta_true = 2.1
    theta_true = np.array([S0_true, beta_true])

    # Independent variable data
    distance = np.array([191, 263, 309])  # distance in metres

    mu = fit_model(distance, theta_true)

    # Store the data of each simulation
    S0_est = []
    beta_est = []
    S0_error = []
    beta_error = []
    cost_min = []

    # Initialize random generator for reproducibility
    rng = np.random.default_rng(6870)

    for iteration in range(1, SIMULATIONS + 1):

        if iteration % 100 == 0:
            print(f'Running iteration {iteration}')

        particles = poisson.rvs(mu, random_state=rng)

        fitter = likefit.Poisson(distance, particles, fit_model)
        fitter.fit(seed=theta_true)

        theta_est = fitter.get_estimators()
        S0_est.append(theta_est[0])
        beta_est.append(theta_est[1])

        errors = fitter.get_errors()
        S0_error.append(errors[0])
        beta_error.append(errors[1])

        cost_min.append(fitter.get_chi_square())


    # Transform lists to numpy arrays
    S0_est = np.asarray(S0_est)
    beta_est = np.asarray(beta_est)
    S0_error = np.asarray(S0_error)
    beta_error = np.asarray(beta_error)
    cost_min = np.asarray(cost_min)
    
    # Performance of the parameter estimators
    bias_S0, bias_S0_error = danatools.get_bias(S0_est, S0_true)
    print(f'Bias S₀ estimator: {bias_S0:.3f} ± {bias_S0_error:.3f}')
    bias_beta, bias_beta_error = danatools.get_bias(beta_est, beta_true)
    print(f'Bias β estimator: {bias_beta:.3f} ± {bias_beta_error:.3f}')

    plot_histo_S0(S0_est, S0_true)
    plot_histo_beta(beta_est, beta_true)

    # Coverage of the confidence intervals
    coverage_S0, coverage_S0_error = danatools.get_coverage(S0_est-S0_error, S0_est+S0_error, S0_true)
    print(f'Coverage S0 interval: ({coverage_S0 * 100:.1f} ± {coverage_S0_error * 100:.1f})%')
    coverage_beta, coverage_beta_error = danatools.get_coverage(beta_est-beta_error, beta_est+beta_error, beta_true)
    print(f'Coverage β interval: ({coverage_beta * 100:.1f} ± {coverage_beta_error * 100:.1f})%')

    plot_S0_errors(S0_error)
    plot_beta_errors(beta_error)

    # Plot deviance and pvalue distributions
    ndof = len(distance) - len(theta_true)
    plot_deviance(cost_min, ndof)

    pvalues = chi2.sf(cost_min, df=ndof)
    plot_pvalue(pvalues)


# Fit model
def fit_model(x, par):
    x0 = 250  # Reference distance in metres
    return par[0] * np.power(x/x0, -par[1])


def plot_histo_S0(S0_est, S0_true):
    fig, ax = plt.subplots(tight_layout=True)
    ax.set_xlabel('$\hat{S}_0$')
    ax.set_ylabel('Events')
    histo, bin_edges = np.histogram(S0_est, bins=50, range=[10, 30])
    bin_centers = (bin_edges[0:-1] + bin_edges[1:]) / 2
    ax.plot(bin_centers, histo, drawstyle='steps-mid', color='tab:blue')
    S0_est_mean = mean(S0_est)
    ax.axvline(S0_est_mean, ls='--', color='tab:orange')
    ax.text(S0_est_mean, 0.3, ' $\hat{E}(\hat{S}_0)$', color='tab:orange', transform=ax.get_xaxis_transform())
    # ax.text(mean_S0, 0.5, f' {mean_S0:.2f} ± {sigma_S0_mean:.2f} ',
    # transform=ax1.get_xaxis_transform(), fontsize="small")
    ax.axvline(S0_true, ls='--', color='tab:gray')
    ax.text(S0_true, 0.3, '$S_0$ ', color='tab:gray', transform=ax.get_xaxis_transform(), horizontalalignment='right')

    danatools.savefigs("fit_perf_S0")


def plot_histo_beta(beta_est, beta_true):
    # TODO: check the peak in the histogram of β at ≈2.5 (Markus comment)
    fig, ax = plt.subplots(tight_layout=True)
    ax.set_xlabel('$\hat{β}$')
    ax.set_ylabel('Events')
    histo, bin_edges = np.histogram(beta_est, bins=50, range=[0, 5])
    bin_centers = (bin_edges[0:-1] + bin_edges[1:]) / 2
    ax.plot(bin_centers, histo, drawstyle='steps-mid', color='tab:blue')
    beta_est_mean = mean(beta_est)
    ax.axvline(beta_est_mean, ls='--', color='tab:orange')
    ax.text(beta_est_mean, 0.3, ' $\hat{E}(\hat{β})$', color='tab:orange', transform=ax.get_xaxis_transform())
    # ax.text(media_beta, 0.5, f' {media_beta:.3f} ± {sigma_beta_mean:.3f}',
    # transform=ax2.get_xaxis_transform(), fontsize="small")
    ax.axvline(beta_true, ls='--', color='tab:gray')
    ax.text(beta_true, 0.3, 'β ', color='tab:gray', transform=ax.get_xaxis_transform(), horizontalalignment='right')

    danatools.savefigs("fit_perf_beta")


def plot_S0_errors(S0_errors):
    fig, ax = plt.subplots(tight_layout=True)
    ax.set_xlabel('Error $S_0$')
    ax.set_ylabel('Events')
    histo, bin_edges = np.histogram(S0_errors, bins=50, range=[1.5, 3.5])
    bin_centers = (bin_edges[0:-1] + bin_edges[1:]) / 2
    ax.plot(bin_centers, histo, drawstyle='steps-mid', color='tab:blue')

    # print('Plotting S0 errors histogram in ' + filename)
    danatools.savefigs("fit_perf_dS0")


def plot_beta_errors(beta_errors):
    fig, ax = plt.subplots(tight_layout=True)
    ax.set_xlabel('Error β')
    ax.set_ylabel('Events')
    histo, bin_edges = np.histogram(beta_errors, bins=50, range=[0.4, 0.9])
    bin_centers = (bin_edges[0:-1] + bin_edges[1:]) / 2
    ax.plot(bin_centers, histo, drawstyle='steps-mid', color='tab:blue')

    # print('Plotting beta errors histogram in ' + filename)
    danatools.savefigs("fit_perf_dbeta")


def plot_deviance(deviances, ndof):
    fig, ax = plt.subplots(tight_layout=True)
    ax.set_xlabel('$J_{\min}$')
    ax.set_ylabel('Events')
    histo, bin_edges = np.histogram(deviances, bins=50, range=[0, 0.5])
    bin_centers = (bin_edges[0:-1] + bin_edges[1:]) / 2
    ax.plot(bin_centers, histo, drawstyle='steps-mid', color='tab:blue', label="Data")
    x = np.linspace(0, 0.5, 256)
    bin_width = bin_edges[1] - bin_edges[0]
    y = chi2.pdf(x, ndof) * SIMULATIONS * bin_width
    ax.plot(x, y, ls='--', color='tab:orange', label="PDF")
    ax.legend()

    danatools.savefigs("fit_perf_deviance")


def plot_pvalue(pvalues):
    fig, ax = plt.subplots(tight_layout=True)
    ax.set_xlabel('pvalue')
    ax.set_ylabel('Events')
    bins = 10
    ax.hist(pvalues, bins=bins, range=[0, 1], histtype='step')
    ax.set_ylim(bottom=0)

    mean_entries = len(pvalues) / bins
    ax.axhline(mean_entries, ls='--', color='tab:orange')

    # print('Plotting pvalues histogram in ' + filename)
    danatools.savefigs("fit_perf_pvalue")


# Run starts here
if __name__ == "__main__":
    main()
