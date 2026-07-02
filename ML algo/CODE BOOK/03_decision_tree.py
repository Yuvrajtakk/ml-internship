# ==============================================================
# 03_decision_tree.py
# DECISION TREE — Learning Rules by Asking Yes/No Questions
# ==============================================================
# WHAT    : Splits data by asking binary questions on features.
#           Each split tries to make the resulting groups as
#           "pure" as possible (all same class in each group).
#
# ANALOGY : A doctor diagnosing a patient:
#           "Is temperature > 38°C? → Yes → Is heartrate > 100? →
#            Yes → Likely infected. No → Maybe just heat stress."
#           That's a decision tree. A series of if/else rules.
#
# GINI IMPURITY (the math):
#           Measures how mixed a group is.
#           Gini = 1 - Σ(probability of each class)²
#           Gini = 0.0 → perfectly pure (all one class) ← WANT THIS
#           Gini = 0.5 → perfectly mixed (50/50 split)  ← BAD
#           At each node, the tree picks the split that gives
#           the biggest DROP in Gini.
#
# WHEN    : Works on any data type. Naturally handles non-linear
#           boundaries. Produces human-readable IF/ELSE rules.
#
# STRENGTH: Results are fully explainable. No scaling needed.
#           Can model complex non-linear boundaries.
#
# WEAKNESS: Deep trees memorize training data (overfitting).
#           Small data changes can totally change the tree.
# ==============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)

# ── DATASET: 100 Trucks, 3 Features ───────────────────────────
# Feature 1: Engine temperature (°C)
# Feature 2: Vibration level (G-force)
# Feature 3: Oil pressure (PSI)
# Label: 0 = Healthy | 1 = Failing

n = 50
healthy_X = np.column_stack([
    np.random.uniform(70, 95, n),     # Normal engine temp
    np.random.uniform(0.05, 0.38, n), # Low vibration
    np.random.uniform(35, 55, n)      # Good oil pressure
])
failing_X = np.column_stack([
    np.random.uniform(105, 140, n),   # Overheating
    np.random.uniform(0.55, 0.95, n), # High vibration
    np.random.uniform(15, 32, n)      # Low oil pressure
])

X = np.vstack([healthy_X, failing_X])
y = np.array([0]*n + [1]*n)

# No scaling needed — Decision Trees split on raw thresholds,
# not distances. Scaling doesn't change the split points.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

feature_names = ['Engine Temp (°C)', 'Vibration (G)', 'Oil Pressure (PSI)']
class_names   = ['Healthy', 'Failing']

# ── MODEL: Controlled Depth vs Deep (Overfit) ─────────────────
# max_depth=3: 3 levels of questions. Generalizes well.
# max_depth=None: unlimited depth — memorizes every training point.

tree_good  = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
tree_overfit = DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=42)

tree_good.fit(X_train, y_train)
tree_overfit.fit(X_train, y_train)

acc_good_train   = accuracy_score(y_train, tree_good.predict(X_train))
acc_good_test    = accuracy_score(y_test,  tree_good.predict(X_test))
acc_overfit_train = accuracy_score(y_train, tree_overfit.predict(X_train))
acc_overfit_test  = accuracy_score(y_test,  tree_overfit.predict(X_test))

print("=" * 60)
print("  DECISION TREE — RESULTS")
print("=" * 60)
print(f"  Training trucks : {len(X_train)}  |  Test trucks: {len(X_test)}")
print()
print(f"  DEPTH=3 (controlled):")
print(f"    Train accuracy: {acc_good_train:.2%}")
print(f"    Test  accuracy: {acc_good_test:.2%}")
print(f"    Gap: {acc_good_train - acc_good_test:.2%}  ← small gap = good generalization")
print()
print(f"  DEPTH=unlimited (overfit):")
print(f"    Train accuracy: {acc_overfit_train:.2%}  ← memorized every training point")
print(f"    Test  accuracy: {acc_overfit_test:.2%}  ← fails on unseen data")
print(f"    Gap: {acc_overfit_train - acc_overfit_test:.2%}  ← large gap = overfitting")
print()
print(f"  Feature importances (which feature the tree relied on most):")
for fname, imp in zip(feature_names, tree_good.feature_importances_):
    bar = '█' * int(imp * 40)
    print(f"    {fname:<25}: {imp:.3f}  {bar}")
print("=" * 60)

# Print the actual rules the tree learned
print("\n  Rules learned (Depth=3 tree reads like IF/THEN):")
print("  → Run visualization to see the full flowchart")
print("=" * 60)

# Predict one new truck
new_truck = np.array([[118, 0.71, 25]])   # Overheating, high vibration, low pressure
pred = tree_good.predict(new_truck)[0]
prob = tree_good.predict_proba(new_truck)[0]
print(f"\n  New truck (Temp=118, Vib=0.71, Oil=25):")
print(f"  → Healthy probability: {prob[0]:.2%}")
print(f"  → Failing probability: {prob[1]:.2%}")
print(f"  → Prediction: {'Failing' if pred == 1 else 'Healthy'}")
print("=" * 60)

# ── VISUALIZATION ─────────────────────────────────────────────
# We visualize using only 2 features (Temp + Vibration) for the
# 2D boundary plot — easier to understand visually.
X_2d       = X[:, :2]          # First two features only for the boundary map
X_train_2d = X_train[:, :2]
X_test_2d  = X_test[:, :2]

tree_2d = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
tree_2d.fit(X_train_2d, y_train)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# LEFT: Decision boundary for the 2D tree (Temp vs Vibration)
x_min, x_max = X_2d[:, 0].min() - 5, X_2d[:, 0].max() + 5
y_min, y_max = X_2d[:, 1].min() - 0.05, X_2d[:, 1].max() + 0.05
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.3),np.arange(y_min, y_max, 0.005))
Z = tree_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
axes[0].contourf(xx, yy, Z, alpha=0.25, cmap='bwr')
axes[0].scatter(X_2d[y==0, 0], X_2d[y==0, 1],
                color='blue', edgecolor='k', s=45, alpha=0.7, label='Healthy (0)')
axes[0].scatter(X_2d[y==1, 0], X_2d[y==1, 1],
                color='red', edgecolor='k', s=45, alpha=0.7, label='Failing (1)')
axes[0].set_xlabel('Engine Temperature (°C)')
axes[0].set_ylabel('Vibration (G-force)')
axes[0].set_title(f'Decision Boundary (Depth=3)\nAxis-aligned rectangular regions')
axes[0].legend(fontsize=8)

# MIDDLE: Tree flowchart — the actual rules
plot_tree(tree_good,feature_names=feature_names,class_names=class_names,filled=True,fontsize=7,ax=axes[1])
axes[1].set_title('Decision Tree Flowchart (Depth=3)\nBlue = Healthy tendency | Red = Failing tendency')

# RIGHT: Overfitting comparison — accuracy bars
labels  = ['Depth=3\n(Controlled)', 'Depth=None\n(Overfit)']
train_accs = [acc_good_train, acc_overfit_train]
test_accs  = [acc_good_test,  acc_overfit_test]

x_pos = np.arange(len(labels))
w     = 0.3
b1 = axes[2].bar(x_pos - w/2, train_accs, w, color='steelblue', edgecolor='k', label='Train accuracy')
b2 = axes[2].bar(x_pos + w/2, test_accs,  w, color='tomato',    edgecolor='k', label='Test accuracy')
for bar in b1:
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,f'{bar.get_height():.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in b2:
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,f'{bar.get_height():.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[2].set_ylim(0.5, 1.08)
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(labels)
axes[2].set_ylabel('Accuracy')
axes[2].set_title('Overfitting Demo\nTrain vs Test accuracy gap')
axes[2].legend()
axes[2].grid(True, alpha=0.3, axis='y')

plt.suptitle('DECISION TREE — Truck Health Classifier (100 Trucks, 3 Features)',fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
