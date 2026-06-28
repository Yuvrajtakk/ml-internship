# ==============================================================
# 06_svm.py
# SUPPORT VECTOR MACHINE — 3 Kernel Variants
# ==============================================================
# WHAT    : Finds the widest possible "street" (margin) between
#           two classes. The boundary is defined ONLY by the
#           closest points to it — called "support vectors".
#           Every other point is completely ignored once training
#           is done. Adding 1000 far-away points changes nothing.
#
# ANALOGY : Draw the widest possible road between two groups.
#           The road edges graze only the people sitting closest
#           to the middle. Moving anyone far from the road
#           changes nothing. Moving the closest people shifts
#           the entire road. Those closest people = support vectors.
#
# WHY MAXIMIZE THE MARGIN?
#   A wider margin means the boundary is as far as possible from
#   both classes. This gives more room for error on new data.
#   A narrow boundary close to training points = fragile.
#
# C PARAMETER (misclassification penalty):
#   High C → strict, tries to get everything right, small margin,
#             more support vectors, risk of overfitting.
#   Low C  → tolerant, allows some wrong points, wider margin,
#             better generalization on unseen data.
#
# KERNELS — the trick for non-linear data:
#   The kernel trick mathematically maps features into a higher
#   dimension where a straight boundary DOES work, without
#   ever actually computing the full high-dimensional space.
#
# VARIANT 1 — LINEAR  (kernel='linear'):
#   Draws a flat hyperplane boundary. You can see the actual
#   margin lines (dashed) and the support vectors (circled).
#   Use when: data is cleanly separable by a straight line.
#
# VARIANT 2 — RBF  (kernel='rbf', Radial Basis Function):
#   Creates circular/blob-shaped boundaries around clusters.
#   gamma: how tightly the boundary wraps around each point.
#     High gamma → tight wrap → risk of overfitting
#     Low gamma  → loose wrap → risk of underfitting
#   Use when: one class wraps around or surrounds the other.
#
# VARIANT 3 — POLYNOMIAL  (kernel='poly'):
#   Creates curved polynomial-shaped boundaries.
#   degree: how many bends/curves the boundary can make.
#     degree=2 → one curve | degree=3 → two curves
#   Use when: boundary has a smooth curved shape.
#
# SCALING IS MANDATORY: SVM computes distances between points.
#   Unscaled features with different ranges distort distances.
#   Always apply StandardScaler before fitting SVM.
#
# WHEN    : Small-to-medium datasets with clear structure.
#           Works well even when features outnumber samples.
#
# STRENGTH: Maximum margin = confident predictions.
#           Kernel trick handles complex non-linear patterns.
#
# WEAKNESS: Slow on large datasets (>100k samples).
#           Hard to interpret — no simple rules to explain.
#           Very sensitive to C and kernel hyperparameters.
# ==============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

np.random.seed(42)

# ── DATASET 1: Linear — Two well-separated truck clusters ──────
# Healthy: low speed + low vibration
# Failing: high speed + high vibration
# A straight line can separate these — Linear kernel is enough.
n = 55
healthy_X = np.column_stack([
    np.random.uniform(30, 60, n),
    np.random.uniform(0.05, 0.40, n)
])
failing_X = np.column_stack([
    np.random.uniform(78, 115, n),
    np.random.uniform(0.58, 0.98, n)
])
X_linear = np.vstack([healthy_X, failing_X])
y_linear  = np.array([0]*n + [1]*n)

# ── DATASET 2: RBF — Concentric circles ───────────────────────
# Inner circle = Class 0, Outer ring = Class 1.
# A straight line CANNOT separate inner from outer.
# RBF kernel creates a circular boundary that can.
X_rbf, y_rbf = make_circles(n_samples=220, factor=0.4, noise=0.09, random_state=42)

# ── DATASET 3: Polynomial — Two crescent moons ────────────────
# Two interleaved crescent/moon shapes.
# Neither a straight line nor a circle separates these.
# A polynomial degree=3 boundary can curve around both moons.
X_poly, y_poly = make_moons(n_samples=220, noise=0.14, random_state=42)

# ── SCALE ALL DATASETS ────────────────────────────────────────
scaler_l = StandardScaler();  X_lin_s  = scaler_l.fit_transform(X_linear)
scaler_r = StandardScaler();  X_rbf_s  = scaler_r.fit_transform(X_rbf)
scaler_p = StandardScaler();  X_poly_s = scaler_p.fit_transform(X_poly)

# ── MODELS ────────────────────────────────────────────────────
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_rbf    = SVC(kernel='rbf',    C=1.0, gamma='scale', random_state=42)
svm_poly   = SVC(kernel='poly',   C=1.0, degree=3, coef0=1, random_state=42)

svm_linear.fit(X_lin_s,  y_linear)
svm_rbf.fit(X_rbf_s,    y_rbf)
svm_poly.fit(X_poly_s,  y_poly)

acc_l = accuracy_score(y_linear, svm_linear.predict(X_lin_s))
acc_r = accuracy_score(y_rbf,    svm_rbf.predict(X_rbf_s))
acc_p = accuracy_score(y_poly,   svm_poly.predict(X_poly_s))

print("=" * 65)
print("  SVM — 3 KERNEL VARIANTS")
print("=" * 65)
print(f"  {'Kernel':<12} | {'Accuracy':>9} | {'Support Vectors':>17} | Best for")
print("  " + "-" * 60)
print(f"  {'LINEAR':<12} | {acc_l:>9.2%} | {str(svm_linear.n_support_):>17} | Straight-line separation")
print(f"  {'RBF':<12} | {acc_r:>9.2%} | {str(svm_rbf.n_support_):>17} | Circular/blob patterns")
print(f"  {'POLYNOMIAL':<12} | {acc_p:>9.2%} | {str(svm_poly.n_support_):>17} | Curved moon-like patterns")
print("=" * 65)
print()
print("  KEY INSIGHT: Support vectors are the ONLY training points")
print("  that matter. Remove any non-support-vector point and the")
print("  boundary stays exactly the same. This is why SVM is robust.")
print()
print("  HOW TO CHOOSE A KERNEL:")
print("  1. Always start with Linear. Fast and interpretable.")
print("  2. If Linear is poor, try RBF. Works for most problems.")
print("  3. Use Polynomial if you know the boundary is polynomial-shaped.")
print("  4. Always scale your data first — SVM needs it.")
print("=" * 65)

# Predictions
new_l = scaler_l.transform([[72, 0.52]])
new_r = scaler_r.transform([[0.2, 0.2]])
new_p = scaler_p.transform([[0.5, 0.0]])
print(f"\n  Linear — Truck (Speed=72, Vib=0.52): {'Failing' if svm_linear.predict(new_l)[0]==1 else 'Healthy'}")
print(f"  RBF    — Point (0.2, 0.2):            Class {svm_rbf.predict(new_r)[0]}")
print(f"  Poly   — Point (0.5, 0.0):            Class {svm_poly.predict(new_p)[0]}")
print("=" * 65)


# ── HELPER: Draw boundary + support vectors ────────────────────
def draw_svm_boundary(ax, model, X_s, y, title,
                      xlabel='Feature 1', ylabel='Feature 2',
                      show_margin=False, new_pt=None):
    x_min = X_s[:, 0].min() - 0.5;  x_max = X_s[:, 0].max() + 0.5
    y_min = X_s[:, 1].min() - 0.5;  y_max = X_s[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.025),
                         np.arange(y_min, y_max, 0.025))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.25, cmap='bwr')
    ax.scatter(X_s[y==0, 0], X_s[y==0, 1], color='blue', edgecolor='k',
               s=55, alpha=0.75, label='Class 0 (Healthy)')
    ax.scatter(X_s[y==1, 0], X_s[y==1, 1], color='red', edgecolor='k',
               s=55, alpha=0.75, label='Class 1 (Failing)')

    # Circle the support vectors — the only points that define the boundary
    sv = model.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=200, facecolors='none',
               edgecolors='black', linewidths=2.5, zorder=5, label='Support Vectors ⭕')

    # For Linear kernel only: draw the hyperplane + margin lanes
    if show_margin and model.kernel == 'linear':
        w         = model.coef_[0]
        slope     = -w[0] / w[1]
        intercept = -model.intercept_[0] / w[1]
        margin    = 1 / w[1]          # Vertical distance from hyperplane to margin lane
        x_range   = np.linspace(x_min, x_max, 200)
        hp        = slope * x_range + intercept   # Hyperplane line

        ax.plot(x_range, hp, 'k-', linewidth=2.5, label='Hyperplane (decision boundary)')
        ax.plot(x_range, hp + margin, 'k--', linewidth=1.8, label='Margin lanes (maximize width)')
        ax.plot(x_range, hp - margin, 'k--', linewidth=1.8)
        ax.fill_between(x_range, hp - margin, hp + margin,
                        alpha=0.12, color='gold', label='Margin zone (maximize this!)')

    if new_pt is not None:
        ax.scatter(new_pt[0], new_pt[1], color='green', marker='*',
                   s=350, zorder=6, label='New point')

    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, loc='upper left')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.2)


# ── VISUALIZATION: 1 row, 3 subplots ──────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
plt.subplots_adjust(wspace=0.28)

draw_svm_boundary(
    axes[0], svm_linear, X_lin_s, y_linear,
    title=(f'LINEAR Kernel  |  C=1.0\n'
           f'Accuracy: {acc_l:.1%}  |  Support Vectors: {sum(svm_linear.n_support_)}\n'
           f'Straight hyperplane + maximum margin street\n'
           f'Gold zone = the margin (wider = better)'),
    xlabel='Speed (scaled)',
    ylabel='Vibration (scaled)',
    show_margin=True,
    new_pt=new_l[0]
)

draw_svm_boundary(
    axes[1], svm_rbf, X_rbf_s, y_rbf,
    title=(f'RBF Kernel  |  C=1.0, gamma=scale\n'
           f'Accuracy: {acc_r:.1%}  |  Support Vectors: {sum(svm_rbf.n_support_)}\n'
           f'Circular boundary wraps around inner cluster\n'
           f'A straight line CANNOT separate these two classes'),
    new_pt=new_r[0]
)

draw_svm_boundary(
    axes[2], svm_poly, X_poly_s, y_poly,
    title=(f'POLYNOMIAL Kernel  |  C=1.0, degree=3\n'f'Accuracy: {acc_p:.1%}  |  Support Vectors: {sum(svm_poly.n_support_)}\n'f'Curved boundary separates crescent moon shapes\n'f'Neither linear nor circular would work here'),
    new_pt=new_p[0]
)

plt.suptitle('SVM — 3 Kernels: Same Maximum Margin Idea, Different Boundary Shapes',fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
