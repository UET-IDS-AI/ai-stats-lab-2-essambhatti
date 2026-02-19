"""
AI Mathematical Tools – Probability & Random Variables

Instructions:
- Implement ALL functions.
- Do NOT change function names or signatures.
- Do NOT print inside functions.
- You may use: math, numpy, matplotlib.
"""

import math
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Part 1 — Probability Foundations
# ============================================================

def probability_union(PA, PB, PAB):
    """
    P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
    """
    return PA + PB - PAB


def conditional_probability(PAB, PB):
    """
    P(A|B) = P(A ∩ B) / P(B)
    """
    return PAB / PB


def are_independent(PA, PB, PAB, tol=1e-9):
    """
    True if:
        |P(A ∩ B) - P(A)P(B)| < tol
    """
    if abs(PAB - (PA*PB)) < tol:
        return True
    return False

def bayes_rule(PBA, PA, PB):
    """
    P(A|B) = P(B|A)P(A) / P(B)
    """
    return (PBA * PA) / PB

# ============================================================
# Part 2 — Bernoulli Distribution
# ============================================================

def bernoulli_pmf(x, theta):
    """
    f(x, theta) = theta^x (1-theta)^(1-x)
    """
    return (theta ** x) * ((1 - theta) ** (1 - x))

def bernoulli_theta_analysis(theta_values):
    """
    Returns:
        (theta, P0, P1, is_symmetric)
    """

    returns = []
    for i in theta_values:
        p_0 = 1 - i
        p_1 = i
        returns.append([i, p_0, p_1, p_0 == p_1 ])
    return returns

    


# ============================================================
# Part 3 — Normal Distribution
# ============================================================

def normal_pdf(x, mu, sigma):
    """
    Normal PDF:
        1/(sqrt(2π)σ) * exp(-(x-μ)^2 / (2σ^2))
    """
    return (1/((2*math.pi)**(1/2)*sigma)) * math.exp( - (x - mu)**2 / (2 * sigma**2))


def normal_histogram_analysis(mu_values,
                              sigma_values,
                              n_samples=10000,
                              bins=30):
    """
    For each (mu, sigma):

    Return:
        (
            mu,
            sigma,
            sample_mean,
            theoretical_mean,
            mean_error,
            sample_variance,
            theoretical_variance,
            variance_error
        )
    """
    results = []
    
    for mu in mu_values:
        for sigma in sigma_values:
            
            samples = np.random.normal(mu, sigma, n_samples)
            
            sample_mean = np.mean(samples)
            sample_variance = np.var(samples, ddof=1)  # unbiased estimator
            
            theoretical_mean = mu
            theoretical_variance = sigma ** 2
            
            mean_error = abs(sample_mean - theoretical_mean)
            variance_error = abs(sample_variance - theoretical_variance)
            
            results.append((
                mu,
                sigma,
                sample_mean,
                theoretical_mean,
                mean_error,
                sample_variance,
                theoretical_variance,
                variance_error
            ))
    
    return results
        



# ============================================================
# Part 4 — Uniform Distribution
# ============================================================

def uniform_mean(a, b):
    """
    (a + b) / 2
    """
    return (a+b)/2


def uniform_variance(a, b):
    """
    (b - a)^2 / 12
    """
    return ((b-a)**2)/12


def uniform_histogram_analysis(a_values,
                               b_values,
                               n_samples=10000,
                               bins=30):
    """
    For each (a, b):

    Return:
        (
            a,
            b,
            sample_mean,
            theoretical_mean,
            mean_error,
            sample_variance,
            theoretical_variance,
            variance_error
        )
    """

    results = []

    for a, b in zip(a_values, b_values):
        samples = np.random.uniform(a, b, n_samples)

        sample_mean = np.mean(samples)
        sample_variance = np.var(samples)

        theoretical_mean = (a + b) / 2
        theoretical_variance = ((b - a) ** 2) / 12

        mean_error = abs(sample_mean - theoretical_mean)
        variance_error = abs(sample_variance - theoretical_variance)

        results.append(
            (
                a,
                b,
                sample_mean,
                theoretical_mean,
                mean_error,
                sample_variance,
                theoretical_variance,
                variance_error
            )
        )

        plt.figure()
        plt.hist(samples, bins=bins, density=True)
        plt.title(f"Uniform Distribution (a={a}, b={b})")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.show()

    return results

if __name__ == "__main__":
    print("Implement all required functions.")
