import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def truncated_power_law(x, a, b, c):
    """Truncated power-law function: a * x^(-b) * exp(-x/c)"""
    return a * np.power(x, -b) * np.exp(-x / c)

def r_squared(y_true, y_pred):
    """Compute coefficient of determination (R²)"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def fit_truncated_power_law(degrees,
                             title="Truncated Power-Law Fit",
                             xlabel="Degree",
                             ylabel="Frequency",
                             show_plot=True,
                             figsize=(8, 6),
                             color_data="black",
                             color_fit="darkred"):
    """
    Fit a truncated power-law to a degree distribution and optionally plot the result.

    Args:
        degrees (array-like): Degree values (not yet counted).
        title (str): Title of the plot.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        show_plot (bool): Whether to display the plot.
        figsize (tuple): Size of the figure.
        color_data (str): Color for scatter data.
        color_fit (str): Color for the fitted curve.

    Returns:
        dict: Dictionary with fit parameters and R².
    """
    degrees = np.asarray(degrees)
    values, counts = np.unique(degrees, return_counts=True)

    # Fit
    popt, _ = curve_fit(truncated_power_law, values, counts, maxfev=10000)
    fit_values = truncated_power_law(values, *popt)
    r2 = r_squared(counts, fit_values)

    if show_plot:
        plt.figure(figsize=figsize)
        plt.scatter(values, counts, label="Data", color=color_data)
        plt.plot(values, fit_values, label=f"Fit (R² = {r2:.2f})", color=color_fit)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return {
        "parameters": {"a": popt[0], "b": popt[1], "c": popt[2]},
        "r_squared": r2,
        "x": values,
        "y": counts,
        "fit": fit_values
    }