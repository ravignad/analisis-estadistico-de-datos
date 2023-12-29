# Cuadrados mínimos lineal con la librería sda

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sda


def get_model_matrix(x):
     return np.column_stack([x ** 0, x ** 1, x ** 2])


input_file = 'beta.dat'
data = pd.read_csv(input_file)

model_matrix = get_model_matrix(data["x"])
fit_result = sda.linear_least_squares(model_matrix, data["beta"], data["dbeta"])
print("Fit result")
print(fit_result)

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
