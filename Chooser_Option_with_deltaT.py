import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
import time

S0 = 100.0
K = 100.0
T_annual = 1.0
U_annual = 0.5 #（range 0-1）
r_annual = 0.05
sigma = 0.20

def bsm_price(S, K, T, r, sigma, option_type='call'):
    if T == 0: return max(0, S - K) if option_type == 'call' else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)
    return price

def chooser_price_analytical(S0, K, T_annual, U_annual, r_annual, sigma):
    K_prime = K * np.exp(-r_annual * (T_annual - U_annual))
    call_T = bsm_price(S0, K, T_annual, r_annual, sigma, 'call')
    put_U = bsm_price(S0, K_prime, U_annual, r_annual, sigma, 'put')
    return call_T + put_U

def chooser_pricer_binomial(n, S0, K, T_annual, U_annual, r_annual, sigma):
    dt = T_annual / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    r_factor = np.exp(r_annual * dt)
    q = (r_factor - d) / (u - d)
    U_step = int(U_annual / dt)

    stock_tree = np.zeros((n + 1, n + 1))
    for t in range(n + 1):
        for i in range(t + 1):
            stock_tree[i, t] = S0 * (u ** (t - i)) * (d ** i)

    call_tree = np.zeros((n + 1, n + 1))
    put_tree = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        call_tree[i, n] = max(0, stock_tree[i, n] - K)
        put_tree[i, n] = max(0, K - stock_tree[i, n])
    for t in range(n - 1, U_step - 1, -1):
        for i in range(t + 1):
            call_tree[i, t] = (q * call_tree[i, t + 1] + (1 - q) * call_tree[i + 1, t + 1]) / r_factor
            put_tree[i, t] = (q * put_tree[i, t + 1] + (1 - q) * put_tree[i + 1, t + 1]) / r_factor

    chooser_tree = np.zeros((n + 1, n + 1))
    for i in range(U_step + 1):
        chooser_tree[i, U_step] = max(call_tree[i, U_step], put_tree[i, U_step])
    for t in range(U_step - 1, -1, -1):
        for i in range(t + 1):
            chooser_tree[i, t] = (q * chooser_tree[i, t + 1] + (1 - q) * chooser_tree[i + 1, t + 1]) / r_factor
    return chooser_tree[0, 0]

n_range = list(range(10, 201, 10)) + list(range(250, 1001, 50))
binomial_prices = []

start_time = time.time()
analytical_price = chooser_price_analytical(S0, K, T_annual, U_annual, r_annual, sigma)
end_time = time.time()
print(
    f"BSM Analytical Solution (Theoretical Value): {analytical_price:.5f} (Calculation time: {end_time - start_time:.4f}s)")
print("-" * 75)
print(f"{'Steps (n)':<12} | {'Binomial Price':<18} | {'Theoretical Value (BSM)':<25} | {'Absolute Error':<15}")
print("-" * 75)

for n in n_range:
    start_time = time.time()
    price = chooser_pricer_binomial(n, S0, K, T_annual, U_annual, r_annual, sigma)
    end_time = time.time()
    binomial_prices.append(price)
    error = abs(price - analytical_price)
    print(f"{n:<12} | {price:<18.5f} | {analytical_price:<25.5f} | {error:<15.5f} (Time: {end_time - start_time:.4f}s)")

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(12, 7))

plt.plot(n_range, binomial_prices, marker='o', linestyle='-', color='dodgerblue',
         label='Binomial Model Price (Approximation)')

plt.axhline(y=analytical_price, color='red', linestyle='--', linewidth=2,
            label=f'BSM Analytical Solution (Theoretical Value): {analytical_price:.4f}')

plt.title('Convergence of Binomial Model to BSM Model for a Chooser Option', fontsize=16)
plt.xlabel('Number of Binomial Steps (n) - Model Granularity', fontsize=12)
plt.ylabel('European Chooser Option Price', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True)
plt.tight_layout()

plt.text(n_range[-1], analytical_price, '  ← Target of Price Convergence', color='red', va='center', ha='left')
plt.text(n_range[2], binomial_prices[2] + 0.1, 'Larger n, smaller Δt, price approaches true value →', color='darkgreen',
         va='bottom', ha='left', fontsize=11)

plt.show()