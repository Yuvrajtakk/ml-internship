# ==============================================================
# 08_evaluation_metrics.py
# EVALUATION METRICS — Accuracy, Precision, Recall, F1
# ==============================================================
# WHY THIS MATTERS:
#   After training a model, you need to measure HOW GOOD it is.
#   Accuracy feels like the obvious answer, but it can completely
#   mislead you on imbalanced datasets. This file shows why,
#   and introduces three better metrics to use alongside it.
#
# THE ACCURACY TRAP:
#   You have 100 trucks. 95 are healthy. 5 are about to fail.
#   A model that always predicts "Healthy" — no matter what —
#   gets 95% accuracy. But it misses EVERY failing truck.
#   Those trucks break down on the highway. Accuracy said 95%.
#   Reality: the model is completely useless.
#
# THE FOUR BUILDING BLOCKS (from the confusion matrix):
#   TP (True Positive)  → Predicted Failing, Actually Failing  ✓ Correct alarm
#   TN (True Negative)  → Predicted Healthy, Actually Healthy  ✓ Correct silence
#   FP (False Positive) → Predicted Failing, Actually Healthy  ✗ False alarm
#   FN (False Negative) → Predicted Healthy, Actually Failing  ✗ Missed failure
#
# THE FOUR METRICS:
#   Accuracy  = (TP + TN) / Total
#               How often is the model right overall?
#               ⚠ Misleading on imbalanced data.
#
#   Precision = TP / (TP + FP)
#               Of all trucks flagged as Failing, how many truly were?
#               Punishes false alarms. Use when false alarms are costly
#               (e.g., unnecessary surgery, spam filter killing real emails).
#
#   Recall    = TP / (TP + FN)
#               Of all actually Failing trucks, how many did we catch?
#               Punishes missed failures. Use when misses are dangerous
#               (e.g., cancer screening, truck engine failures).
#
#   F1 Score  = 2 × (Precision × Recall) / (Precision + Recall)
#               Harmonic mean of Precision and Recall.
#               Use when you cannot afford to sacrifice either one.
#               Always lower than the simple average of the two.
#
# PRECISION vs RECALL TRADE-OFF:
#   Lowering the decision threshold → catches more Failing trucks
#   (Recall goes up) but also flags more Healthy ones (Precision drops).
#   Raising the threshold → fewer false alarms (Precision goes up)
#   but misses more real failures (Recall drops).
#   F1 rewards finding a balance between the two.
#
# MULTI-CLASS (like MNIST with 10 digit classes):
#   Each metric is computed per class, then averaged.
#   average='weighted': classes with more samples pull more weight.
#   classification_report() shows everything at once cleanly.
# ==============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score, confusion_matrix, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(42)


# =============================================================
# PART 1: THE ACCURACY TRAP — Why accuracy alone is dangerous
# =============================================================
print("=" * 65)
print("  PART 1: THE ACCURACY TRAP")
print("=" * 65)
print("  Scenario: 100 trucks. 90 healthy, 10 failing.")
print("  A 'lazy' model that always predicts Healthy:")
print()

y_actual_imbalanced = np.array([0]*90 + [1]*10)        # 90 healthy, 10 failing
y_lazy_model        = np.array([0]*100)                 # Always predicts Healthy

lazy_acc  = accuracy_score(y_actual_imbalanced, y_lazy_model)
lazy_rec  = recall_score(y_actual_imbalanced, y_lazy_model, zero_division=0)
lazy_prec = precision_score(y_actual_imbalanced, y_lazy_model, zero_division=0)
lazy_f1   = f1_score(y_actual_imbalanced, y_lazy_model, zero_division=0)

print(f"  Accuracy  : {lazy_acc:.0%}  ← Looks great! But...")
print(f"  Recall    : {lazy_rec:.0%}   ← ZERO failing trucks caught")
print(f"  Precision : {lazy_prec:.0%}   ← Flagged nobody, so no false alarms")
print(f"  F1 Score  : {lazy_f1:.0%}   ← Punishes the zero recall severely")
print()
print("  The lazy model caught ZERO failing trucks while scoring 90%.")
print("  This is why you always check Recall when failures are costly.")
print("=" * 65)


# =============================================================
# PART 2: MANUAL WALKTHROUGH — TP, TN, FP, FN by hand
# =============================================================
print()
print("=" * 65)
print("  PART 2: MANUAL WALKTHROUGH — 10 trucks, step by step")
print("=" * 65)

y_actual    = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
y_predicted = np.array([0, 0, 0, 1, 0, 0, 1, 0, 1, 1])

print("  Truck  | Actual  | Predicted | Result")
print("  " + "-" * 44)
for i, (act, pred) in enumerate(zip(y_actual, y_predicted)):
    if act == 1 and pred == 1: result = "TP — Caught a real failure ✓"
    elif act == 0 and pred == 0: result = "TN — Correctly left alone  ✓"
    elif act == 0 and pred == 1: result = "FP — False alarm (Healthy flagged as Failing)"
    else: result = "FN — MISSED a real failure ✗"
    a_str = 'Healthy' if act == 0 else 'Failing'
    p_str = 'Healthy' if pred == 0 else 'Failing'
    print(f"  {i+1:5d}  | {a_str:<7} | {p_str:<9} | {result}")

cm = confusion_matrix(y_actual, y_predicted)
TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]

acc  = accuracy_score(y_actual, y_predicted)
prec = precision_score(y_actual, y_predicted)
rec  = recall_score(y_actual, y_predicted)
f1   = f1_score(y_actual, y_predicted)

print()
print(f"  Counts: TP={TP}  TN={TN}  FP={FP}  FN={FN}")
print()
print(f"  Accuracy  = (TP+TN)/Total = ({TP}+{TN})/10 = {acc:.2f}")
print(f"  Precision = TP/(TP+FP)   = {TP}/({TP}+{FP})  = {prec:.2f}")
print(f"  Recall    = TP/(TP+FN)   = {TP}/({TP}+{FN})  = {rec:.2f}")
print(f"  F1 Score  = 2×(P×R)/(P+R)= 2×({prec:.2f}×{rec:.2f})/({prec:.2f}+{rec:.2f}) = {f1:.2f}")
print("=" * 65)


# =============================================================
# PART 3: REAL MODEL — Logistic Regression on 120 Trucks
# =============================================================
print()
print("=" * 65)
print("  PART 3: REAL MODEL — Logistic Regression on 120 Trucks")
print("=" * 65)

n = 55
healthy_X = np.column_stack([np.random.uniform(30, 65, n),
                              np.random.uniform(0.05, 0.40, n)])
failing_X = np.column_stack([np.random.uniform(75, 115, n),
                              np.random.uniform(0.58, 0.98, n)])
# 10 ambiguous trucks that are hard to classify
ambig_X   = np.column_stack([np.random.uniform(63, 78, 10),
                              np.random.uniform(0.38, 0.60, 10)])

X_real = np.vstack([healthy_X, failing_X, ambig_X])
y_real = np.concatenate([[0]*n, [1]*n, np.random.choice([0,1], 10)])

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_real)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_real, test_size=0.2, random_state=42
)

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)
y_test_prob = model.predict_proba(X_test)[:, 1]

acc_r  = accuracy_score(y_test, y_test_pred)
prec_r = precision_score(y_test, y_test_pred, zero_division=0)
rec_r  = recall_score(y_test, y_test_pred, zero_division=0)
f1_r   = f1_score(y_test, y_test_pred, zero_division=0)
cm_r   = confusion_matrix(y_test, y_test_pred)

print(f"  Training trucks: {len(X_train)}  |  Test trucks: {len(X_test)}")
print()
print(f"  Default threshold (0.5):")
print(f"    Accuracy  : {acc_r:.2%}")
print(f"    Precision : {prec_r:.2%}  ← of all flagged as Failing, how many truly were?")
print(f"    Recall    : {rec_r:.2%}  ← of all truly Failing, how many did we catch?")
print(f"    F1 Score  : {f1_r:.2%}  ← balance of precision and recall")
print()

# Custom threshold demo
y_paranoid = (y_test_prob >= 0.3).astype(int)
prec_p = precision_score(y_test, y_paranoid, zero_division=0)
rec_p  = recall_score(y_test, y_paranoid, zero_division=0)
f1_p   = f1_score(y_test, y_paranoid, zero_division=0)
print(f"  Paranoid threshold (0.3) — flags more, misses fewer:")
print(f"    Precision : {prec_p:.2%}  ← drops (more false alarms)")
print(f"    Recall    : {rec_p:.2%}  ← rises (fewer failures missed)")
print(f"    F1 Score  : {f1_p:.2%}")
print()

# sklearn's classification_report: one-liner for everything
print("  sklearn classification_report (the full picture at once):")
print()
print(classification_report(y_test, y_test_pred,
                             target_names=['Healthy (0)', 'Failing (1)']))
print("=" * 65)


# =============================================================
# VISUALIZATION
# =============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(hspace=0.40, wspace=0.32)

# TOP-LEFT: Confusion Matrix heatmap for manual example (10 trucks)
cm_manual = confusion_matrix(y_actual, y_predicted)
axes[0, 0].imshow(cm_manual, cmap='Blues')
cell_labels = [['TN', 'FP'], ['FN', 'TP']]
cell_colors = [['Correct', 'False alarm'],
               ['Missed', 'Correct']]
for i in range(2):
    for j in range(2):
        val   = cm_manual[i, j]
        label = f"{val}\n{cell_labels[i][j]}"
        color = 'white' if val >= 2 else 'black'
        axes[0, 0].text(j, i, label, ha='center', va='center',
                        fontsize=10, fontweight='bold', color=color)
axes[0, 0].set_xticks([0, 1])
axes[0, 0].set_yticks([0, 1])
axes[0, 0].set_xticklabels(['Predicted: Healthy', 'Predicted: Failing'], fontsize=8)
axes[0, 0].set_yticklabels(['Actual: Healthy', 'Actual: Failing'], fontsize=8)
axes[0, 0].set_title('Confusion Matrix (10 trucks)\nTP|FP|FN|TN breakdown', fontsize=9, fontweight='bold')

# TOP-RIGHT: 4 metrics bar chart for the real model
metrics_names  = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
metrics_values = [acc_r, prec_r, rec_r, f1_r]
bar_colors     = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

bars = axes[0, 1].bar(metrics_names, metrics_values, color=bar_colors, edgecolor='k', width=0.55, linewidth=0.8)
for bar, val in zip(bars, metrics_values):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.025, f'{val:.2f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[0, 1].set_ylim(0, 1.2)
axes[0, 1].axhline(1.0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
axes[0, 1].set_ylabel('Score', fontsize=8)
axes[0, 1].set_title(f'All 4 Metrics (threshold=0.5)\n{len(X_test)} test trucks, Logistic Regression', fontsize=9, fontweight='bold')
axes[0, 1].grid(True, alpha=0.2, axis='y')
axes[0, 1].tick_params(labelsize=7.5)

# BOTTOM-LEFT: Precision-Recall trade-off curve across thresholds
thresholds = np.arange(0.05, 0.98, 0.04)
precs_t, recs_t, f1s_t = [], [], []
for t in thresholds:
    preds_t = (y_test_prob >= t).astype(int)
    precs_t.append(precision_score(y_test, preds_t, zero_division=1))
    recs_t.append(recall_score(y_test, preds_t, zero_division=0))
    f1s_t.append(f1_score(y_test, preds_t, zero_division=0))

axes[1, 0].plot(thresholds, precs_t, color='darkorange', linewidth=2, label='Precision', marker='o', markersize=3)
axes[1, 0].plot(thresholds, recs_t,  color='steelblue',  linewidth=2, label='Recall', marker='s', markersize=3)
axes[1, 0].plot(thresholds, f1s_t,   color='green', linewidth=1.8, linestyle='-.', label='F1 Score')
axes[1, 0].axvline(0.5, color='gray', linestyle='--', linewidth=1.2, alpha=0.6, label='Default (0.5)')
axes[1, 0].axvline(0.3, color='purple', linestyle=':', linewidth=1.2, alpha=0.6, label='Paranoid (0.3)')
axes[1, 0].set_xlabel('Decision Threshold', fontsize=8)
axes[1, 0].set_ylabel('Score', fontsize=8)
axes[1, 0].set_title('Precision-Recall Trade-off\nLower threshold → catch more (Recall↑, Precision↓)', fontsize=8.5, fontweight='bold')
axes[1, 0].legend(fontsize=7)
axes[1, 0].grid(True, alpha=0.25)
axes[1, 0].set_ylim(-0.05, 1.1)
axes[1, 0].tick_params(labelsize=7)

# BOTTOM-RIGHT: The Accuracy Trap — visual comparison
trap_metrics = ['Accuracy', 'Recall', 'Precision', 'F1 Score']
lazy_values  = [lazy_acc, lazy_rec, lazy_prec, lazy_f1]
trap_colors  = ['#4C72B0', '#C44E52', '#55A868', '#DD8452']

bars_trap = axes[1, 1].bar(trap_metrics, lazy_values, color=trap_colors, edgecolor='k', width=0.55, linewidth=0.8)
for bar, val in zip(bars_trap, lazy_values):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.025, f'{val:.0%}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[1, 1].set_ylim(0, 1.2)
axes[1, 1].set_ylabel('Score', fontsize=8)
axes[1, 1].set_title('THE ACCURACY TRAP ⚠\n"Always predict Healthy" on imbalanced data\nAccuracy=90% but Recall=0% — dangerous!', fontsize=8.5, fontweight='bold')
axes[1, 1].axhline(1.0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
axes[1, 1].grid(True, alpha=0.2, axis='y')
axes[1, 1].tick_params(labelsize=7.5)

plt.suptitle('EVALUATION METRICS — Accuracy, Precision, Recall, F1',
             fontsize=12, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
