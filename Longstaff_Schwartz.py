import numpy as np
from sklearn.linear_model import LinearRegression

S0 = 100
K = 100
r = 0.05
sigma = 0.2
T = 1.0
U = 0.5
n_steps = 50
n_paths = 10000

dt = T / n_steps
discount = np.exp(-r * dt)
U_step = int(U / dt)

np.random.seed(42)
Z = np.random.randn(n_paths, n_steps)
S = np.zeros((n_paths, n_steps + 1))
S[:, 0] = S0

for t in range(1, n_steps + 1):
    S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1])
def lsm_option_value(S_paths, K, r, dt, option_type='call'):
    n_paths, n_steps = S_paths.shape
    cashflows = np.zeros_like(S_paths)
    if option_type == 'call':
        cashflows[:, -1] = np.maximum(S_paths[:, -1] - K, 0)
    else:
        cashflows[:, -1] = np.maximum(K - S_paths[:, -1], 0)
    for t in range(n_steps - 2, -1, -1):
        itm = (cashflows[:, t + 1] > 0)
        X = S_paths[itm, t]
        Y = cashflows[itm, t + 1] * discount
        if len(X) == 0:
            continue
        X = X.reshape(-1, 1)
        model = LinearRegression().fit(X, Y)
        continuation = model.predict(X)
        exercise = (
            np.maximum(S_paths[itm, t] - K, 0)
            if option_type == 'call'
            else np.maximum(K - S_paths[itm, t], 0)
        )

        exercise_indices = np.where(exercise > continuation)[0]
        full_indices = np.where(itm)[0][exercise_indices]
        cashflows[full_indices, t] = exercise[exercise_indices]
        cashflows[full_indices, t + 1:] = 0
    discounts = np.exp(-r * dt * np.arange(cashflows.shape[1]))
    discounted_values = cashflows * discounts
    return np.mean(np.max(discounted_values, axis=1))

S_U = S[:, U_step:]
call_val = lsm_option_value(S_U, K, r, dt, option_type='call')
put_val = lsm_option_value(S_U, K, r, dt, option_type='put')

chooser_price = np.exp(-r * U) * np.mean(np.maximum(call_val, put_val))

print(f"American-style Chooser Option Price (via LSM): {chooser_price:.4f}")