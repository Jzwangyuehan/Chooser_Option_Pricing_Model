import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt

S0 = 100
K = 100
r = 0.05
sigma = 0.2
T = 1
U = 0.5
n_paths = 500
# seed = 42
# np.random.seed(seed)

def bs_price(S, K, r, sigma, tau, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    if option_type == 'call':
        price = S * si.norm.cdf(d1) - K * np.exp(-r * tau) * si.norm.cdf(d2)
    else:
        price = K * np.exp(-r * tau) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)
    return price
Z = np.random.randn(n_paths)
S_U = S0 * np.exp((r - 0.5 * sigma**2) * U + sigma * np.sqrt(U) * Z)
tau = T - U

call_values = bs_price(S_U, K, r, sigma, tau, option_type='call')
put_values = bs_price(S_U, K, r, sigma, tau, option_type='put')
chooser_payoff = np.maximum(call_values, put_values)
discounted_payoff = np.exp(-r * U) * chooser_payoff

MC_estimate_chooser = discounted_payoff.mean()
rolling_average_chooser = np.cumsum(discounted_payoff) / (np.arange(n_paths) + 1)

print(f"Monte Carlo Chooser Option Price: {MC_estimate_chooser:.4f}")
print(f"Rolling Average Estimate: {rolling_average_chooser.mean():.4f}")


plt.figure(figsize=(10, 6))
plt.plot(rolling_average_chooser, color='green', label='Rolling Average - Chooser Option')
plt.axhline(MC_estimate_chooser, color='black', linestyle='--', label='Final MC Estimate')
plt.xlabel("Number of Simulations")
plt.ylabel("Discounted Payoff Estimate")
plt.title("Monte Carlo Rolling Estimate of Chooser Option")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

