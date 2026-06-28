# ==============================================================
# 04_random_forest.py
# RANDOM FOREST — An Ensemble of Decision Trees
# ==============================================================
# WHAT    : Builds many Decision Trees, each trained on a random
#           subset of the data and features, then takes a majority
#           vote across all trees for the final prediction.
#
# ANALOGY : You're unsure about a medical diagnosis, so you get
#           a second opinion. Then a third. Then a fourth.
#           Each doctor sees slightly different test results.
#           You go with whatever most of them agree on.
#           That's a Random Forest — a committee of Decision Trees.
#
# WHY IT BEATS A SINGLE TREE:
#           A single deep tree memorizes training data (overfits).
#           Each tree in the forest makes different mistakes because
#           they see different data and features. Averaging out many
#           different mistakes → fewer mistakes overall.
#
# TWO SOURCES OF RANDOMNESS (what makes each tree different):
#   1. Bootstrap sampling: each tree trains on a random sample
#      of rows (with replacement, so some rows appear twice).
#   2. Feature randomness: at each split, only √(n_features)
#      randomly chosen features are considered.
#
# OUT-OF-BAG (OOB) SCORE:
#           The ≈37% of rows NOT selected for a tree's bootstrap
#           sample serve as a free built-in validation set.
#           oob_score_ is the accuracy on these left-out samples.
#           No separate test set needed to get an honest estimate.
#
# WHEN    : Almost always better than a single Decision Tree.
#           Works well out-of-the-box with minimal tuning.
#
# STRENGTH: Handles noise well. Gives feature importances.
#           OOB score means less need for train/test split.
#
# WEAKNESS: Slower than a single tree. Harder to interpret
#           (you can't read 100 trees like you can read one).
# ==============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)

# ── DATASET: 120 Trucks, 3 Features ───────────────────────────
# Feature 1: Speed (km/h)
# Feature 2: Vibration (G-force)
# Feature 3: Engine temperature (°C)
# Label: 0 = Healthy | 1 = Failing
# We add noisy middle-ground trucks to make it harder.

n = 50
healthy_X = np.column_stack([
    np.random.uniform(30, 62, n),
    np.random.uniform(0.05, 0.38, n),
    np.random.uniform(70, 94, n)
])
failing_X = np.column_stack([
    np.random.uniform(78, 115, n),
    np.random.uniform(0.58, 0.97, n),
    np.random.uniform(108, 142, n)
])
# 20 ambiguous trucks in the overlap zone (harder to classify)
ambig_X = np.column_stack([
    np.random.uniform(60, 82, 20),
    np.random.uniform(0.36, 0.62, 20),
    np.random.uniform(92, 112, 20)
])
ambig_y = np.random.choice([0, 1], 20, p=[0.5, 0.5])

X = np.vstack([healthy_X, failing_X, ambig_X])
y = np.concatenate([[0]*n, [1]*n, ambig_y])

feature_names = ['Speed (km/h)', 'Vibration (G)', 'Engine Temp (°C)']
class_names   = ['Healthy', 'Failing']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── MODELS: Single Tree vs Forest ─────────────────────────────
single_tree = DecisionTreeClassifier(max_depth=None, random_state=42)
forest      = RandomForestClassifier(
    n_estimators=100,      # 100 trees in the committee
    max_features='sqrt',   # At each split, consider √3 ≈ 2 features randomly
    oob_score=True,        # Use the left-out rows for free validation
    random_state=42
)

single_tree.fit(X_train, y_train)
forest.fit(X_train, y_train)

acc_tree_train  = accuracy_score(y_train, single_tree.predict(X_train))
acc_tree_test   = accuracy_score(y_test,  single_tree.predict(X_test))
acc_forest_train = accuracy_score(y_train, forest.predict(X_train))
acc_forest_test  = accuracy_score(y_test,  forest.predict(X_test))

print("=" * 60)
print("  RANDOM FOREST — RESULTS")
print("=" * 60)
print(f"  Training trucks : {len(X_train)}  |  Test trucks: {len(X_test)}")
print()
print("  SINGLE TREE (unlimited depth):")
print(f"    Train accuracy : {acc_tree_train:.2%}  ← memorized training data")
print(f"    Test  accuracy : {acc_tree_test:.2%}  ← poor on unseen data")
print(f"    Overfitting gap: {acc_tree_train - acc_tree_test:.2%}")
print()
print("  RANDOM FOREST (100 trees):")
print(f"    Train accuracy : {acc_forest_train:.2%}")
print(f"    Test  accuracy : {acc_forest_test:.2%}  ← much better generalization")
print(f"    OOB  accuracy  : {forest.oob_score_:.2%}  ← free estimate using left-out rows")
print(f"    Overfitting gap: {acc_forest_train - acc_forest_test:.2%}  ← much smaller")
print()
print("  Feature importances (which features the forest relied on most):")
for fname, imp in zip(feature_names, forest.feature_importances_):
    bar = '█' * int(imp * 50)
    print(f"    {fname:<25}: {imp:.3f}  {bar}")
print("=" * 60)

# Predict a new truck
new_truck = np.array([[85, 0.72, 119]])
pred  = forest.predict(new_truck)[0]
probs = forest.predict_proba(new_truck)[0]
print(f"\n  New truck (Speed=85, Vib=0.72, Temp=119):")
print(f"  → Healthy vote: {probs[0]:.0%} of 100 trees")
print(f"  → Failing vote: {probs[1]:.0%} of 100 trees")
print(f"  → Final prediction: {'Failing' if pred == 1 else 'Healthy'}")
print("=" * 60)

# ── VISUALIZATION ─────────────────────────────────────────────
fig = plt.figure(figsize=(22, 11))
plt.subplots_adjust(hspace=0.35, wspace=0.3)

# TOP-LEFT: Decision boundary using Speed + Vibration (2D slice)
ax1 = fig.add_subplot(2, 3, (1, 4))   # Takes up the left column
forest_2d = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
forest_2d.fit(X_train[:, :2], y_train)

x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
y_min, y_max = X[:, 1].min() - 0.05, X[:, 1].max() + 0.05
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),
                     np.arange(y_min, y_max, 0.01))
Z = forest_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax1.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
ax1.scatter(X[y==0, 0], X[y==0, 1], color='blue', edgecolor='black', s=80, alpha=0.8, label='Healthy (0)')
ax1.scatter(X[y==1, 0], X[y==1, 1], color='red', edgecolor='black', s=80, alpha=0.8, label='Failing (1)')
ax1.set_xlabel('Speed (km/h)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Vibration (G-force)', fontsize=11, fontweight='bold')
ax1.set_title(f'RANDOM FOREST — 100 Trees, Smarter Together Than Any Single Tree\nOOB Accuracy: {forest_2d.oob_score_:.1%}  |  Smooth boundary = committee vote', 
              fontsize=12, fontweight='bold', pad=15)
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(True, alpha=0.2, linestyle='--')

# TOP-RIGHT: 4 individual trees from inside the forest (each is different!)
for i in range(4):
    position = [2, 3, 5, 6][i]
    ax = fig.add_subplot(2, 3, position)
    plot_tree(forest.estimators_[i],
              feature_names=feature_names,
              class_names=class_names,
              filled=True, fontsize=6, ax=ax, rounded=True)
    ax.set_title(f'Tree {i+1} (one of 100)\nNotice: different questions!', fontsize=10, fontweight='bold')

# Feature importance bar chart in position 6 (bottom right)
ax_imp = fig.add_subplot(2, 3, 6)
importances = forest.feature_importances_
colors_imp  = ['steelblue', 'darkorange', 'green']
bars = ax_imp.bar(feature_names, importances, color=colors_imp, edgecolor='black', linewidth=1.5, alpha=0.8)
for bar, imp in zip(bars, importances):
    height = bar.get_height()
    ax_imp.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{imp:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax_imp.set_ylabel('Importance Score', fontsize=11, fontweight='bold')
ax_imp.set_title('Feature Importances\n(which features the forest leaned on most)', fontsize=11, fontweight='bold')
ax_imp.set_ylim(0, max(importances) * 1.15)
ax_imp.tick_params(axis='x', labelsize=9)
ax_imp.tick_params(axis='y', labelsize=9)
ax_imp.grid(True, alpha=0.2, axis='y', linestyle='--')

plt.suptitle('RANDOM FOREST — 100 Trees, Smarter Together Than Any Single Tree',
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
