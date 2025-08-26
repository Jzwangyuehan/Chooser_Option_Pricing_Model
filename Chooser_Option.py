import matplotlib.pyplot as plt
import numpy as np
import os

S0 = 115
u = 1.2
d = 0.8
r = 0.05
K = 100
T = 5
U = 10

def build_price_tree(S0, u, d, T):
    stock_tree = np.zeros((T + 1, T + 1))
    for t in range(T + 1):
        for i in range(t + 1):
            stock_tree[i, t] = S0 * (u ** (t - i)) * (d ** i)
    return stock_tree
stock_tree = np.round((build_price_tree(S0, u, d, T)),2)

def chooser_pricer(S0, u, d, r, K, T, U):
    q = (1 + r - d) / (u - d)
    U = min(U, T)
    stock_tree = build_price_tree(S0, u, d, T)
    call_payoff = np.maximum(stock_tree[:, T] - K, 0)
    put_payoff = np.maximum(K - stock_tree[:, T], 0)
    call_tree = np.zeros((T + 1, T + 1))
    put_tree = np.zeros((T + 1, T + 1))
    call_tree[:, T] = call_payoff
    put_tree[:, T] = put_payoff
    for t in range(T - 1, U - 1, -1):
        for i in range(t + 1):
            call_tree[i, t] = 1 / (1 + r) * (q * call_tree[i, t + 1] + (1 - q) * call_tree[i + 1, t + 1])
            put_tree[i, t] = 1 / (1 + r) * (q * put_tree[i, t + 1] + (1 - q) * put_tree[i + 1, t + 1])
    chooser_tree = np.zeros((T + 1, T + 1))
    for i in range(U + 1):
        chooser_tree[i, U] = max(float(call_tree[i, U]), float(put_tree[i, U]))
    for t in range(U - 1, -1, -1):
        for i in range(t + 1):
            chooser_tree[i, t] = 1 / (1 + r) * (q * chooser_tree[i, t + 1] + (1 - q) * chooser_tree[i + 1, t + 1])
    return chooser_tree[0, 0]
Chooser_price = chooser_pricer(S0, u, d, r, K, T, U)

def option_pricer(payoff_vector, u, d, r, T):
    q = (1 + r - d) / (u - d)
    option_tree = np.zeros((T + 1, T + 1))
    option_tree[:, T] = payoff_vector
    for t in range(T - 1, -1, -1):
        for i in range(t + 1):
            option_tree[i, t] = 1 / (1 + r) * (q * option_tree[i, t + 1] + (1 - q) * option_tree[i + 1, t + 1])
    return option_tree[0, 0]
Call_payoff_vector = np.maximum(stock_tree[:,T] - K, 0)
Put_payoff_vector = np.maximum(K - stock_tree[:,T], 0)
call_price = option_pricer(Call_payoff_vector, u, d, r, T)
put_price = option_pricer(Put_payoff_vector, u, d, r, T)

print(f"When T={T}; U={U}; S0={S0}; K={K}:")
print(f"Chooser  Fair Price (t=0): {Chooser_price:.4f}")
print(f"European Call Price (t=0): {call_price:.4}")
print(f"European Put  Price (t=0): {put_price:.4}")

print(f"\n-----------Option Prices Across Different T when U = {U}-----------\n")
T_range = list(range(1, T + 1))
U_fixed = U
def compute_prices_over_T(S0, u, d, r, K, T_range, U_fixed):
    chooser_prices = []
    call_prices = []
    put_prices = []
    q = (1 + r - d) / (u - d)
    for t in T_range:
        stock_tree_T = build_price_tree(S0, u, d, t)
        call_payoff = np.maximum(stock_tree_T[:, t] - K, 0)
        put_payoff = np.maximum(K - stock_tree_T[:, t], 0)

        chooser = chooser_pricer(S0, u, d, r, K, t, U_fixed)
        call = option_pricer(call_payoff, u, d, r, t)
        put = option_pricer(put_payoff, u, d, r, t)

        chooser_prices.append(chooser)
        call_prices.append(call)
        put_prices.append(put)
    return chooser_prices, call_prices, put_prices

chooser_prices, call_prices, put_prices = compute_prices_over_T(S0, u, d, r, K, T_range, U_fixed)

# ===================================================================Graph
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

for idx, t in enumerate(T_range):
    print(f"When T={t:<3}:  Chooser = {chooser_prices[idx]:>6.2f}  "
          f"|  Call = {call_prices[idx]:>6.2f}  "
          f"|  Put = {put_prices[idx]:>6.2f}  |")
axs[0, 0].plot(T_range, chooser_prices, marker='o', label='Chooser Option Price', color='r')
axs[0, 0].plot(T_range, call_prices, marker='s', label='Call Option Price', color='b')
axs[0, 0].plot(T_range, put_prices, marker='^', label='Put Option Price', color='g')
axs[0, 0].set_xlabel('Time to Maturity T')
axs[0, 0].set_ylabel('Option Price')
axs[0, 0].set_title(f'Option Prices vs T (U fixed at {U_fixed})')
axs[0, 0].legend()
axs[0, 0].grid(True)

def generate_U_list(T):
    return list(range(1, T+1, 2))
U_list = generate_U_list(T)

chooser_prices_dict = {}
for U_val in U_list:
    T_valid = [T for T in T_range if T >= U_val]
    if not T_valid:
        continue
    chooser_prices_U, _, _ = compute_prices_over_T(S0, u, d, r, K, T_valid, U_val)
    axs[0, 1].plot(T_valid, chooser_prices_U, label=f'U={U_val}')

for U_val, prices in chooser_prices_dict.items():
    axs[0, 1].plot(T_range, prices, label=f'U={U_val}')
axs[0, 1].plot(T_range, call_prices, marker='o', label='European Call Price', color='b', linestyle='--')
axs[0, 1].plot(T_range, put_prices, marker='o', label='European Put Price', color='g', linestyle='--')
axs[0, 1].set_xlabel('Time to Maturity T')
axs[0, 1].set_ylabel('Option Price')
axs[0, 1].set_title('Chooser Option Price vs T for Different U')
axs[0, 1].set_xticks(T_range)
axs[0, 1].legend()
axs[0, 1].grid(True)

U_range = list(range(1, T + 1))
chooser_prices_over_U = [chooser_pricer(S0, u, d, r, K, T, U_val) for U_val in U_range]
axs[1, 0].plot(U_range, chooser_prices_over_U, marker='o', color='red',label='Chooser Option Price')
axs[1, 0].set_xlabel('Chooser Decision Time U')
axs[1, 0].set_ylabel('Option Price')
axs[1, 0].set_title(f'Chooser Option Price vs U (T fixed at {T})')
axs[1, 0].legend()
axs[1, 0].grid(True)

u_values = np.linspace(1.05, 1.5, 15)
d_values = np.linspace(0.5, 0.95, 15)
chooser_u, call_u, put_u = [], [], []
chooser_d, call_d, put_d = [], [], []
for u_val in u_values:
    stock_tree_u = build_price_tree(S0, u_val, d, T)
    call_payoff_u = np.maximum(stock_tree_u[:, T] - K, 0)
    put_payoff_u = np.maximum(K - stock_tree_u[:, T], 0)
    chooser_u.append(chooser_pricer(S0, u_val, d, r, K, T, U_fixed))
    call_u.append(option_pricer(call_payoff_u, u_val, d, r, T))
    put_u.append(option_pricer(put_payoff_u, u_val, d, r, T))
for d_val in d_values:
    stock_tree_d = build_price_tree(S0, u, d_val, T)
    call_payoff_d = np.maximum(stock_tree_d[:, T] - K, 0)
    put_payoff_d = np.maximum(K - stock_tree_d[:, T], 0)
    chooser_d.append(chooser_pricer(S0, u, d_val, r, K, T, U_fixed))
    call_d.append(option_pricer(call_payoff_d, u, d_val, r, T))
    put_d.append(option_pricer(put_payoff_d, u, d_val, r, T))
axs[1, 1].plot(u_values, chooser_u, 'o-', label='Chooser vs u', color='red')
axs[1, 1].plot(u_values, call_u, 's-', label='Call vs u', color='blue')
axs[1, 1].plot(u_values, put_u, '^-', label='Put vs u', color='green')
axs[1, 1].plot(d_values, chooser_d, 'o--', label='Chooser vs d', color='red')
axs[1, 1].plot(d_values, call_d, 's--', label='Call vs d', color='blue')
axs[1, 1].plot(d_values, put_d, '^--', label='Put vs d', color='green')
axs[1, 1].set_xlabel('Up Factor (u) or Down Factor (d)')
axs[1, 1].set_ylabel('Option Price')
axs[1, 1].set_title('Impact of u and d on Option Prices')
legend_u = axs[1, 1].legend(handles=[
    axs[1, 1].lines[0],
    axs[1, 1].lines[1],
    axs[1, 1].lines[2],
], title="Impact of u", loc='center right', bbox_to_anchor=(1, 0.5))
legend_d = axs[1, 1].legend(handles=[
    axs[1, 1].lines[3],
    axs[1, 1].lines[4],
    axs[1, 1].lines[5],
], title="Impact of d", loc='center left', bbox_to_anchor=(0, 0.5))
axs[1, 1].add_artist(legend_u)
axs[1, 1].grid(True)

# --- SAVE PLOT (place at the very end, after all plotting) ---
import os
out_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "chooser_plots.png")

plt.tight_layout()
plt.savefig(out_path, dpi=200)
print(f"[OK] Saved plot to: {out_path}")
# plt.show()  # 需要弹窗再看就解开；不看就留着保存即可