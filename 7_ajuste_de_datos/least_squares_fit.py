# Pruebo cuadrados mínimos no-lineal de la librería sda

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sda


input_file = 'beta.dat'
data = pd.read_csv(input_file)

xdata = data["x"]
def fit_model(theta):
     return theta[0] + theta[1] * xdata + theta[2] * xdata**2 

cost_function = sda.Least_squares_cost(fit_model, data["beta"], data["dbeta"])

seed = np.array([1, 1, 1])

fit_result = minimize(cost_function, x0=seed) 

# fit_result = sda.least_squares(fit_model, data["beta"], data["dbeta"], seed)

print("Fit result")
print(fit_result)

exit()

# Plot
fig, ax = plt.subplots()
ax.set_xlabel("x")
ax.set_ylabel(r"$\beta$")

ax.errorbar(data["x"], data["beta"], data["dbeta"], ls='none', marker='o', label="Datos")

xmin = 1
xmax = 1.35
xfit = np.linspace(start=xmin, stop=xmax, num=100)
model_matrix_fit = get_model_matrix(xfit)
yfit =  model_matrix_fit  @ fit_result['est']
ax.plot(xfit, yfit, ls='--', label="Ajuste")

yfit_error = sda.fit_errors(model_matrix_fit, fit_result["cova"])
ax.fill_between(xfit, yfit-yfit_error, yfit+yfit_error, color='tab:orange', alpha=0.2)

plt.legend()
plt.tight_layout()
plt.savefig("ajuste_beta.pdf")
