# ==============================================================
# 01_linear_regression.py
# LINEAR REGRESSION — Predicting a Continuous Number
# ==============================================================
# WHAT    : Finds the best straight line through data to predict
#           a numerical output (not a category).
#
# ANALOGY : Trucks driving more km always spend more on fuel.
#           Linear Regression finds the exact formula:
#           Fuel Cost = (₹ per km × distance) + base fee
#
# WHEN    : Your output is a number — cost, salary, temperature,
#           distance, stock price.
#
# STRENGTH: Simple, fast, formula is human-readable.
#           You can explain predictions to anyone.
#
# WEAKNESS: Assumes the relationship is a straight line.
#           If the real pattern curves, this model fails.
#
# FORMULA : y = (w₁×x₁) + (w₂×x₂) + ... + b
#           w = weight (slope), b = bias (intercept)
# ==============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)   # Fixed seed = same random data every run

# ── DATASET: 80 Watsoo Express Trucks ─────────────────────────
# Feature: Distance driven (km)
# Target : Fuel cost (₹)
# True rule the model needs to discover: ₹4.5/km + ₹250 base
# We add noise to simulate real-world variation (traffic, weather, driver style)

n_trucks   = 80
distance   = np.random.uniform(50, 600, n_trucks)        # Random distances 50–600 km
noise      = np.random.normal(0, 120, n_trucks)           # Random variation ±₹120
fuel_cost  = 4.5 * distance + 250 + noise                 # True formula + noise

X = distance.reshape(-1, 1)   # sklearn needs 2D: (80 samples, 1 feature)
y = fuel_cost

# ── SPLIT: 80% Train, 20% Test ────────────────────────────────
# Train on 64 trucks, test on 16 trucks the model has never seen.
# We never let the model see test data during training.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── MODEL ─────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)   # Finds the best slope + intercept from 64 training trucks
y_pred = model.predict(X_test)  # Uses learned formula on 16 unseen test trucks

# ── RESULTS ───────────────────────────────────────────────────
slope     = model.coef_[0]       # Learned ₹/km rate
intercept = model.intercept_     # Learned base fee
rmse      = np.sqrt(mean_squared_error(y_test, y_pred))
# R² (R-squared): 1.0 = perfect predictions, 0.0 = model learned nothing,
# negative = model is worse than just predicting the average
r2        = r2_score(y_test, y_pred)

print("=" * 55)
print("  LINEAR REGRESSION — RESULTS")
print("=" * 55)
print(f"  Learned formula : Fuel = {slope:.2f} × km + {intercept:.2f}")
print(f"  True formula    : Fuel = 4.50 × km + 250.00")
print(f"  Training trucks : {len(X_train)}")
print(f"  Test trucks     : {len(X_test)}")
print(f"  RMSE            : ₹{rmse:.2f}  ← average prediction error")
print(f"  R² Score        : {r2:.4f}  ← 1.0 is perfect")
print("=" * 55)

# Predict fuel cost for a new truck driving 350 km
new_truck_km   = np.array([[350]])
predicted_cost = model.predict(new_truck_km)[0]
print(f"\n  New truck driving 350 km:")
print(f"  → Predicted cost: ₹{predicted_cost:.2f}")
print(f"  → True cost range: ₹{4.5*350+250-120:.0f} to ₹{4.5*350+250+120:.0f}  (with noise)")
print("=" * 55)

# ── VISUALIZATION ─────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# LEFT: All 80 trucks scattered on the map + the learned regression line
sorted_X  = np.sort(X, axis=0)
sorted_preds = model.predict(sorted_X)

ax1.scatter(distance, fuel_cost, alpha=0.55, color='steelblue',
            edgecolor='k', s=55, label='Actual trucks (all 80)')
ax1.plot(sorted_X, sorted_preds, color='red', linewidth=2.5,
         label=f'Learned line: ₹{slope:.1f}/km + ₹{intercept:.0f}')
ax1.scatter([350], [predicted_cost], color='green', marker='*',
            s=300, zorder=6, label=f'New truck (350km) → ₹{predicted_cost:.0f}')
ax1.set_xlabel('Distance Driven (km)')
ax1.set_ylabel('Fuel Cost (₹)')
ax1.set_title('All 80 Trucks: Distance vs Fuel Cost\n(Red line = what the model learned)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# RIGHT: Actual vs Predicted on the 16 test trucks.
# If the model is perfect, every dot lands exactly on the diagonal line.
# Dots above the diagonal = underpredicted. Dots below = overpredicted.
diag_range = [y_test.min(), y_test.max()]
ax2.scatter(y_test, y_pred, color='darkorange', edgecolor='k', s=80, alpha=0.8,
            label='Test truck predictions')
ax2.plot(diag_range, diag_range, 'k--', linewidth=1.5, label='Perfect prediction (diagonal)')
ax2.set_xlabel('Actual Fuel Cost (₹)')
ax2.set_ylabel('Predicted Fuel Cost (₹)')
ax2.set_title(f'Test Set: Actual vs Predicted\nR² = {r2:.3f}  |  RMSE = ₹{rmse:.1f}')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle('LINEAR REGRESSION — Watsoo Express Fuel Cost Predictor',fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
