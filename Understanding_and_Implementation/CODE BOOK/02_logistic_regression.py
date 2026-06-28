# ==============================================================
# 02_logistic_regression.py
# LOGISTIC REGRESSION — Binary Classification
# ==============================================================
# WHAT    : Predicts a CATEGORY (0 or 1), not a number.
#           Underneath it still fits a line, but then squashes
#           that line through a sigmoid function so the output
#           is always a probability between 0.0 and 1.0.
#
# ANALOGY : A doctor doesn't say "your cancer score is 7.3".
#           They say "likely malignant (85% probability)".
#           Logistic Regression gives you that probability.
#
# WHEN    : Binary outcome — spam/not spam, healthy/failing,
#           approved/rejected, fraud/legit.
#
# STRENGTH: Outputs interpretable probabilities, not just labels.
#           You can tune the decision threshold to your risk level.
#
# WEAKNESS: Draws only a straight-line boundary. If the classes
#           are arranged in a curve or circle, it struggles.
#
# SIGMOID  : sigmoid(z) = 1 / (1 + e^(-z))
#            Squashes any number into (0, 1) range.
#            z > 0 → probability > 0.5 → predicted class = 1
# ==============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

np.random.seed(42)

# ── DATASET: 100 Trucks, 2 Features ───────────────────────────
# Feature 1: Speed (km/h)
# Feature 2: Vibration level (G-force)
# Label 0: Healthy truck | Label 1: Failing truck

n = 50
healthy_X = np.column_stack([
    np.random.uniform(30, 65, n),       # Healthy: lower speed
    np.random.uniform(0.05, 0.42, n)    # Healthy: lower vibration
])
failing_X = np.column_stack([
    np.random.uniform(72, 115, n),      # Failing: higher speed
    np.random.uniform(0.55, 0.98, n)    # Failing: higher vibration
])
X = np.vstack([healthy_X, failing_X])
y = np.array([0]*n + [1]*n)

# ── SCALE FEATURES ────────────────────────────────────────────
# Logistic Regression uses gradient descent under the hood.
# Unscaled features with different ranges cause uneven gradient steps.
# Scaling makes both features contribute equally.
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── SPLIT ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ── MODEL ─────────────────────────────────────────────────────
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

y_pred     = model.predict(X_test)
y_prob     = model.predict_proba(X_test)[:, 1]   # Probability of class 1 (Failing)

# ── RESULTS ───────────────────────────────────────────────────
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)

print("=" * 55)
print("  LOGISTIC REGRESSION — DEFAULT THRESHOLD (0.5)")
print("=" * 55)
print(f"  Training trucks : {len(X_train)}")
print(f"  Test trucks     : {len(X_test)}")
print(f"  Accuracy        : {acc:.2%}")
print(f"  Precision       : {prec:.2%}  (of all flagged, how many were truly failing)")
print(f"  Recall          : {rec:.2%}  (of all failing, how many did we catch)")
print(f"  F1 Score        : {f1:.2%}  (balance of precision and recall)")
print("=" * 55)

# ── CUSTOM THRESHOLD DEMO ─────────────────────────────────────
# Default threshold is 0.5: probability ≥ 0.5 → class 1 (Failing)
# For truck safety, we'd rather flag too many than miss a real failure.
# Lowering threshold to 0.3 = more aggressive, higher recall, lower precision.

threshold       = 0.3
y_pred_paranoid = (y_prob >= threshold).astype(int)
prec_p = precision_score(y_test, y_pred_paranoid)
rec_p  = recall_score(y_test, y_pred_paranoid)
f1_p   = f1_score(y_test, y_pred_paranoid)

print(f"\n  CUSTOM THRESHOLD (0.3) — More cautious, flags more trucks")
print("=" * 55)
print(f"  Precision: {prec_p:.2%}  ← drops (more false alarms)")
print(f"  Recall   : {rec_p:.2%}  ← rises (fewer failing trucks missed)")
print(f"  F1 Score : {f1_p:.2%}")
print("=" * 55)

# Predict one specific truck
new_truck = scaler.transform([[78, 0.62]])    # Speed=78, Vibration=0.62
prob_fail = model.predict_proba(new_truck)[0][1]
label     = 'Failing' if prob_fail >= 0.5 else 'Healthy'
print(f"\n  New truck (Speed=78, Vibration=0.62):")
print(f"  → Failing probability: {prob_fail:.2%}")
print(f"  → Prediction (0.5 threshold): {label}")
print(f"  → Prediction (0.3 threshold): {'Failing' if prob_fail >= 0.3 else 'Healthy'}")
print("=" * 55)

# ── VISUALIZATION ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# LEFT: Sigmoid function — the shape that turns any number into 0–1
z   = np.linspace(-8, 8, 300)
sig = 1 / (1 + np.exp(-z))    # sigmoid formula
axes[0].plot(z, sig, color='purple', linewidth=3)
axes[0].axhline(0.5, color='gray', linestyle='--', label='Decision boundary (0.5)')
axes[0].axhline(0.3, color='orange', linestyle='--', label='Custom threshold (0.3)')
axes[0].fill_between(z, sig, 0.5, where=(sig >= 0.5), alpha=0.15, color='red', label='Predicted: Failing')
axes[0].fill_between(z, sig, 0.5, where=(sig < 0.5), alpha=0.15, color='blue', label='Predicted: Healthy')
axes[0].set_xlabel('Linear output z = (w·x + b)')
axes[0].set_ylabel('Probability of Failing')
axes[0].set_title('Sigmoid Function\n(squashes any number into 0–1)')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# MIDDLE: Decision boundary in 2D feature space
x0_min, x0_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
x1_min, x1_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x0_min, x0_max, 0.03),
                     np.arange(x1_min, x1_max, 0.03))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
axes[1].contourf(xx, yy, Z, alpha=0.25, cmap='bwr')
axes[1].scatter(X_scaled[y==0, 0], X_scaled[y==0, 1],
                color='blue', edgecolor='k', s=50, alpha=0.7, label='Healthy (0)')
axes[1].scatter(X_scaled[y==1, 0], X_scaled[y==1, 1],
                color='red', edgecolor='k', s=50, alpha=0.7, label='Failing (1)')
axes[1].scatter(new_truck[0][0], new_truck[0][1], color='green',
                marker='*', s=350, zorder=6, label=f'New truck ({prob_fail:.0%} failing)')
axes[1].set_xlabel('Speed (scaled)')
axes[1].set_ylabel('Vibration (scaled)')
axes[1].set_title(f'Decision Boundary (linear)\nAccuracy: {acc:.1%}')
axes[1].legend(fontsize=8)

# RIGHT: Threshold comparison — how Precision and Recall shift
thresholds = np.arange(0.1, 0.95, 0.05)
precisions = []
recalls    = []
for t in thresholds:
    preds = (y_prob >= t).astype(int)
    # Handle edge case where no positive predictions exist
    if preds.sum() == 0:
        precisions.append(1.0)
        recalls.append(0.0)
    else:
        precisions.append(precision_score(y_test, preds, zero_division=0))
        recalls.append(recall_score(y_test, preds, zero_division=0))

axes[2].plot(thresholds, precisions, color='darkorange', linewidth=2.5, label='Precision')
axes[2].plot(thresholds, recalls, color='steelblue', linewidth=2.5, label='Recall')
axes[2].axvline(0.5, color='gray', linestyle='--', alpha=0.7, label='Default (0.5)')
axes[2].axvline(0.3, color='green', linestyle='--', alpha=0.7, label='Custom (0.3)')
axes[2].set_xlabel('Decision Threshold')
axes[2].set_ylabel('Score')
axes[2].set_title('Precision vs Recall at Different Thresholds\n(trade-off: raising one lowers the other)')
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(0, 1.05)

plt.suptitle('LOGISTIC REGRESSION — Truck Health Classifier',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
