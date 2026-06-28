# ==============================================================
# 05_knn.py
# K-NEAREST NEIGHBORS — Classification by Majority Vote
# ==============================================================
# WHAT    : Stores all training data. At prediction time, finds
#           the K closest points and takes a majority vote.
#           No math during training — all work happens at predict.
#
# ANALOGY : You move to a new city and want to know if your
#           neighbourhood is safe or not. You look at the 5
#           nearest houses. 4 have security cameras, 1 doesn't.
#           Majority says: safe area. That's KNN with K=5.
#
# SCALING IS NON-NEGOTIABLE:
#   Speed ranges 30–115 km/h. Vibration ranges 0.05–0.98 G.
#   Without scaling, a speed difference of 50 km/h contributes
#   2500 to the squared distance. A vibration difference of 0.5 G
#   contributes only 0.25. Vibration might as well not exist.
#   StandardScaler fixes this: z = (x - mean) / std
#   After scaling, both features live on the same numerical scale.
#
# CHOOSING K — THE U-CURVE:
#   K=1    → each training point owns its own territory island.
#            Any noisy outlier creates wrong islands. Overfitting.
#   K=N    → every prediction consults everyone. Always predicts
#            the majority class. Useless. Underfitting.
#   Optimal K → bottom of the U-shaped error curve.
#            Found by cross-validation (test multiple K values,
#            pick the one with lowest validation error).
#
# ODD K PREFERRED: On binary problems, odd K eliminates ties.
#   K=4 → can produce 2 vs 2 tie. K=5 → always has a winner.
#
# WHEN    : Small-to-medium datasets where distances are meaningful.
# STRENGTH: No training time. Naturally handles multi-class.
# WEAKNESS: Slow at prediction on large data. Needs scaled features.
#           "Curse of dimensionality" — degrades with many features.
# ==============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

np.random.seed(42)

# ── DATASET: 80 Trucks, 2 Features ────────────────────────────
# Feature 1: Speed (km/h)      — range 30 to 115
# Feature 2: Vibration (G)     — range 0.05 to 0.98
# These are on very different scales on purpose to show the
# scaling problem clearly in the first visualization.

n = 35
healthy_X = np.column_stack([
    np.random.uniform(30, 63, n),
    np.random.uniform(0.05, 0.40, n)
])
failing_X = np.column_stack([
    np.random.uniform(78, 115, n),
    np.random.uniform(0.58, 0.98, n)
])
# 10 ambiguous trucks in the fuzzy middle zone
ambig_X = np.column_stack([
    np.random.uniform(62, 82, 10),
    np.random.uniform(0.38, 0.62, 10)
])
ambig_y = np.random.choice([0, 1], 10, p=[0.5, 0.5])

X = np.vstack([healthy_X, failing_X, ambig_X])
y = np.concatenate([[0]*n, [1]*n, ambig_y])

# ── SCALE ─────────────────────────────────────────────────────
# fit_transform learns mean + std from data and scales it.
# We call fit_transform on the full dataset here for visualization.
# In a real pipeline you fit only on training data.
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train_s, X_test_s, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ── RESULTS: K=1, K=5, K=11 ───────────────────────────────────
print("=" * 60)
print("  KNN — RESULTS (scaled data)")
print("=" * 60)
print(f"  Training trucks: {len(X_train_s)}  |  Test trucks: {len(X_test_s)}")
print()
print(f"  {'K':>4} | {'Train Acc':>10} | {'Test Acc':>9} | {'Gap':>6} | Notes")
print("  " + "-" * 55)
for k in [1, 3, 5, 7, 9, 11]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_s, y_train)
    tr = accuracy_score(y_train, knn.predict(X_train_s))
    te = accuracy_score(y_test,  knn.predict(X_test_s))
    gap = tr - te
    note = '← overfitting' if gap > 0.1 else ('← stable' if gap < 0.04 else '')
    print(f"  {k:>4} | {tr:>10.2%} | {te:>9.2%} | {gap:>6.2%} | {note}")
print("=" * 60)

# ── OPTIMAL K: Cross-Validation U-Curve ───────────────────────
# Try every odd K from 1 to 25.
# For each K: run 5-fold cross-validation, record mean error.
# The K at the bottom of the U-curve is the optimal one.
k_values  = list(range(1, 26, 2))   # [1, 3, 5, 7, ... 25]
cv_errors = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    # cross_val_score returns accuracy for each fold
    # 1 - accuracy = error rate
    cv_acc = cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy')
    cv_errors.append(1 - cv_acc.mean())

best_k   = k_values[int(np.argmin(cv_errors))]
best_err = min(cv_errors)

print(f"\n  Optimal K from cross-validation: K={best_k}")
print(f"  CV error at K={best_k}: {best_err:.4f}")

# ── PREDICT A NEW TRUCK ───────────────────────────────────────
new_truck_raw    = np.array([[72, 0.52]])     # Unscaled for display
new_truck_scaled = scaler.transform(new_truck_raw)  # Scale using same fit

best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_s, y_train)
pred  = best_knn.predict(new_truck_scaled)[0]
probs = best_knn.predict_proba(new_truck_scaled)[0]

print(f"\n  New truck (Speed=72 km/h, Vibration=0.52 G):")
print(f"  → K={best_k} neighbors consulted")
print(f"  → Healthy vote: {probs[0]:.0%}  |  Failing vote: {probs[1]:.0%}")
print(f"  → Final prediction: {'Failing' if pred == 1 else 'Healthy'}")
print("=" * 60)


# ── HELPER: Draw boundary for any KNN model on given axes ─────
def draw_boundary(ax, model, X_s, y, title, new_pt_scaled=None):
    x_min = X_s[:, 0].min() - 0.4
    x_max = X_s[:, 0].max() + 0.4
    y_min = X_s[:, 1].min() - 0.4
    y_max = X_s[:, 1].max() + 0.4
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.025),
                         np.arange(y_min, y_max, 0.025))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.25, cmap='bwr')
    ax.scatter(X_s[y==0, 0], X_s[y==0, 1], color='blue', edgecolor='k',
               s=35, alpha=0.75, label='Healthy (0)')
    ax.scatter(X_s[y==1, 0], X_s[y==1, 1], color='red', edgecolor='k',
               s=35, alpha=0.75, label='Failing (1)')
    if new_pt_scaled is not None:
        ax.scatter(new_pt_scaled[0], new_pt_scaled[1], color='green',
                   marker='*', s=250, zorder=6, label='New truck')
    ax.set_title(title, fontsize=8.5)
    ax.set_xlabel('Speed (scaled)', fontsize=7.5)
    ax.set_ylabel('Vibration (scaled)', fontsize=7.5)
    ax.legend(fontsize=6, loc='upper left')
    ax.grid(True, alpha=0.2)


# ── VISUALIZATION: 2×3 grid ───────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
plt.subplots_adjust(hspace=0.35, wspace=0.25)

# ── ROW 1: SCALING DEMO ───────────────────────────────────────

# TOP-LEFT: K=5 on UNSCALED data
# Speed dominates — boundary is nearly vertical, Vibration ignored
knn5_unscaled = KNeighborsClassifier(n_neighbors=5)
knn5_unscaled.fit(X, y)   # Fit on raw, unscaled X

x_min_u = X[:, 0].min() - 5;  x_max_u = X[:, 0].max() + 5
y_min_u = X[:, 1].min() - 0.05; y_max_u = X[:, 1].max() + 0.05
xx_u, yy_u = np.meshgrid(np.arange(x_min_u, x_max_u, 0.6),
                          np.arange(y_min_u, y_max_u, 0.01))
Z_u = knn5_unscaled.predict(np.c_[xx_u.ravel(), yy_u.ravel()]).reshape(xx_u.shape)
axes[0, 0].contourf(xx_u, yy_u, Z_u, alpha=0.25, cmap='bwr')
axes[0, 0].scatter(X[y==0, 0], X[y==0, 1], color='blue', edgecolor='k', s=40, alpha=0.75, label='Healthy (0)')
axes[0, 0].scatter(X[y==1, 0], X[y==1, 1], color='red',  edgecolor='k', s=40, alpha=0.75, label='Failing (1)')
axes[0, 0].scatter(new_truck_raw[0][0], new_truck_raw[0][1], color='green', marker='*', s=250, zorder=6, label='New truck')
axes[0, 0].set_title('K=5  |  NO SCALING  ⚠\nSpeed (0–115) crushes Vibration (0–1)\nBoundary is almost vertical — Vibration ignored!', fontsize=9)
axes[0, 0].set_xlabel('Speed km/h — RAW (big numbers)', fontsize=8)
axes[0, 0].set_ylabel('Vibration G — RAW (tiny numbers)', fontsize=8)
axes[0, 0].legend(fontsize=6)
axes[0, 0].grid(True, alpha=0.2)

# TOP-MIDDLE: K=5 on SCALED data — correct boundary
knn5_scaled = KNeighborsClassifier(n_neighbors=5)
knn5_scaled.fit(X_scaled, y)
draw_boundary(axes[0, 1], knn5_scaled, X_scaled, y,
              'K=5  |  WITH SCALING  ✓\nBoth Speed and Vibration contribute equally\nDiagonal boundary = uses both features properly',
              new_pt_scaled=new_truck_scaled[0])

# TOP-RIGHT: U-curve (optimal K)
axes[0, 2].plot(k_values, cv_errors, 'o-', color='steelblue',
                linewidth=2, markersize=6, markerfacecolor='white',
                markeredgewidth=1.5)
axes[0, 2].scatter([best_k], [best_err], color='green', s=150, zorder=5,
                   label=f'Optimal: K={best_k}')
axes[0, 2].axvline(best_k, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
axes[0, 2].set_xlabel('K (number of neighbors)', fontsize=8)
axes[0, 2].set_ylabel('Cross-Validation Error', fontsize=8)
axes[0, 2].set_title(f'U-Curve — Finding Optimal K\nLeft = overfit | Right = underfit | Bottom = K={best_k}', fontsize=8.5)
axes[0, 2].legend(fontsize=6.5)
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].set_xticks(k_values)
axes[0, 2].tick_params(labelsize=7)

# ── ROW 2: K COMPARISON ───────────────────────────────────────
for col, k in enumerate([1, 5, 11]):
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train_s, y_train)
    test_acc = accuracy_score(y_test, knn_k.predict(X_test_s))
    train_acc = accuracy_score(y_train, knn_k.predict(X_train_s))

    if k == 1:
        subtitle = (f'K=1: Every point owns its own island\n'
                    f'Extremely sensitive to noise\n'
                    f'Train: {train_acc:.1%}  |  Test: {test_acc:.1%}  ← Overfit')
    elif k == 5:
        subtitle = (f'K=5: Five-way vote smooths noise\n'
                    f'Outliers get outvoted by neighbors\n'
                    f'Train: {train_acc:.1%}  |  Test: {test_acc:.1%}  ← Good')
    else:
        subtitle = (f'K=11: Larger committee, smoother\n'
                    f'May consult irrelevant far-away points\n'
                    f'Train: {train_acc:.1%}  |  Test: {test_acc:.1%}')

    draw_boundary(axes[1, col], knn_k, X_scaled, y,
                  subtitle, new_pt_scaled=new_truck_scaled[0])

plt.suptitle('KNN — K-Nearest Neighbors (80 Trucks)  |  Scaling + K Selection',
             fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()