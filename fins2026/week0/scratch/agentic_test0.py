import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

dates = pd.date_range("2000-01-31", "2026-05-31", freq="ME")
n = len(dates)

drift = 0.005
vol = 0.03
log_returns = np.random.normal(drift, vol, n)
log_returns[0] = 0
log_price = np.cumsum(log_returns)
price = 100 * np.exp(log_price)

df = pd.DataFrame({"Date": dates, "Price": price.round(6)})
df.to_csv("data/prices.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["Date"], df["Price"])
ax.set_title("Synthetic Monthly Price (Random Walk with Drift)")
ax.set_xlabel("Date")
ax.set_ylabel("Price ($)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("figures/prices.png", dpi=150)
fig.savefig("figures/prices.pdf")
plt.close(fig)

print(f"Generated {n} monthly prices from {dates[0].date()} to {dates[-1].date()}")
print("Saved: data/prices.csv, figures/prices.png, figures/prices.pdf")
