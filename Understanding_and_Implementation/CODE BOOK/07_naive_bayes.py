# ==============================================================
# 07_naive_bayes.py
# NAIVE BAYES — 3 Variants (Gaussian, Multinomial, Bernoulli)
# ==============================================================
# WHAT    : Uses Bayes' Theorem to calculate the probability that
#           a data point belongs to each class, then picks the
#           class with the highest probability.
#
# BAYES' THEOREM (simplified):
#   P(class | features) ∝ P(features | class) × P(class)
#
#   In plain words:
#   "How likely is this class given these features?"
#     = "How often do features like these appear in this class?"
#     × "How common is this class overall in the dataset?"
#
#   The model calculates this for every class, then predicts
#   whichever class has the highest resulting probability.
#
# WHY "NAIVE"?
#   It assumes all features are completely INDEPENDENT of each
#   other — knowing one feature tells you nothing extra about
#   another feature. This is almost never true in real data
#   (speed and vibration DO correlate), but the algorithm still
#   works surprisingly well. The simplification makes it very fast.
#
# ── VARIANT 1: GaussianNB ─────────────────────────────────────
#   ASSUMPTION : Each feature follows a bell curve (Gaussian
#                distribution) within each class.
#   LEARNS     : Mean (μ) and variance (σ²) of each feature
#                per class from training data.
#   PREDICTS   : Plugs new values into the learned bell curves
#                to get P(features | class) for each class.
#   USE WHEN   : Features are continuous decimal numbers
#                (sensor readings, temperatures, speeds, etc.)
#
# ── VARIANT 2: MultinomialNB ──────────────────────────────────
#   ASSUMPTION : Features are counts — non-negative integers.
#   LEARNS     : Probability of each word/feature appearing
#                in each class.
#   PREDICTS   : A report with many "smoke" and "noise" words
#                is more likely to belong to the Failing class.
#   USE WHEN   : Word counts in text, frequency data, any integer
#                count features. Classic for spam filtering.
#   REQUIREMENT: All feature values MUST be ≥ 0 (integers).
#
# ── VARIANT 3: BernoulliNB ────────────────────────────────────
#   ASSUMPTION : Each feature is binary — present (1) or absent (0).
#   LEARNS     : Probability that each binary feature = 1 in each class.
#   PREDICTS   : A truck that failed oil + brake checks is more
#                likely Failing than one that passed everything.
#   USE WHEN   : Checklist data, yes/no flags, presence/absence
#                of characteristics.
#
# WHEN    : Best for text classification, small datasets, or when
#           you need a very fast, simple baseline model.
#
# STRENGTH: Extremely fast training. Works with little data.
#           Handles many features efficiently. Easy to update
#           with new data (online learning).
#
# WEAKNESS: Independence assumption is often wrong. Correlated
#           features hurt accuracy. GaussianNB assumes bell-curve
#           shaped data — skewed distributions reduce performance.
# ==============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)


# =============================================================
# VARIANT 1: GaussianNB — Continuous Sensor Readings
# =============================================================
print("=" * 65)
print("  VARIANT 1: GaussianNB — Continuous Sensor Readings")
print("=" * 65)
print("  Data type : Continuous decimals (speed, vibration)")
print("  Assumption: Each feature is bell-curve shaped per class")
print()

# 100 trucks: [Speed (km/h), Vibration (G-force)]
n = 50
healthy_X = np.column_stack([
    np.random.uniform(30, 65, n),
    np.random.uniform(0.05, 0.40, n)
])
failing_X = np.column_stack([
    np.random.uniform(75, 115, n),
    np.random.uniform(0.58, 0.98, n)
])
X_gauss = np.vstack([healthy_X, failing_X])
y_gauss = np.array([0]*n + [1]*n)

X_g_train, X_g_test, y_g_train, y_g_test = train_test_split(
    X_gauss, y_gauss, test_size=0.2, random_state=42
)

gnb = GaussianNB()
gnb.fit(X_g_train, y_g_train)
g_acc = accuracy_score(y_g_test, gnb.predict(X_g_test))

# gnb.theta_ holds the learned mean per class per feature
# gnb.var_   holds the learned variance per class per feature
print(f"  Training size : {len(X_g_train)} trucks  |  Test accuracy: {g_acc:.2%}")
print()
print("  What GaussianNB learned from training data:")
print(f"  {'Class':<15} {'Speed mean':>12} {'Speed std':>11} {'Vib mean':>10} {'Vib std':>9}")
print("  " + "-" * 60)
for cls, name in [(0, 'Healthy (0)'), (1, 'Failing (1)')]:
    s_mean = gnb.theta_[cls][0];  s_std = np.sqrt(gnb.var_[cls][0])
    v_mean = gnb.theta_[cls][1];  v_std = np.sqrt(gnb.var_[cls][1])
    print(f"  {name:<15} {s_mean:>12.1f} {s_std:>11.1f} {v_mean:>10.3f} {v_std:>9.3f}")

# At prediction: plug new values into both learned bell curves,
# compare probabilities, pick the more likely class.
new_truck_g = np.array([[70, 0.50]])
prob_g = gnb.predict_proba(new_truck_g)[0]
pred_g = gnb.predict(new_truck_g)[0]
print(f"\n  New truck (Speed=70, Vibration=0.50):")
print(f"  → P(Healthy): {prob_g[0]:.2%}  |  P(Failing): {prob_g[1]:.2%}")
print(f"  → Prediction: {'Failing' if pred_g == 1 else 'Healthy'}")
print("=" * 65)


# =============================================================
# VARIANT 2: MultinomialNB — Word Counts in Maintenance Reports
# =============================================================
print()
print("=" * 65)
print("  VARIANT 2: MultinomialNB — Maintenance Report Word Counts")
print("=" * 65)
print("  Data type : Integer counts (non-negative)")
print("  Assumption: Features are word/event frequencies")
print("  Scenario  : Each row = one maintenance report")
print("              Each column = how many times a word appears")
print("  Words tracked: [smoke, noise, leak, normal, smooth, clean]")
print()

# Healthy reports: mechanics write "normal", "smooth", "clean"
# Failing reports: mechanics write "smoke", "noise", "leak"
# ALL values must be non-negative integers for MultinomialNB
healthy_reports = np.array([
    [0, 0, 0, 3, 4, 2], [0, 1, 0, 2, 3, 3], [0, 0, 0, 4, 2, 2],
    [0, 0, 0, 3, 3, 3], [0, 0, 1, 2, 4, 2], [1, 0, 0, 3, 2, 3],
    [0, 1, 0, 4, 3, 1], [0, 0, 0, 3, 4, 3], [0, 0, 0, 2, 3, 4],
    [0, 0, 0, 4, 2, 3], [1, 0, 0, 3, 3, 2], [0, 1, 0, 2, 4, 2],
    [0, 0, 1, 3, 3, 3], [0, 0, 0, 4, 2, 3], [0, 0, 0, 3, 3, 2],
    [1, 0, 0, 2, 4, 3], [0, 1, 0, 3, 3, 2], [0, 0, 0, 4, 2, 4],
    [0, 0, 0, 3, 4, 2], [0, 0, 1, 2, 3, 3], [0, 0, 0, 3, 3, 3],
    [0, 0, 0, 4, 4, 2], [1, 0, 0, 2, 3, 3], [0, 0, 0, 3, 2, 4],
    [0, 1, 0, 4, 3, 2]
])
failing_reports = np.array([
    [3, 2, 1, 0, 0, 0], [4, 3, 2, 0, 1, 0], [2, 4, 1, 0, 0, 0],
    [3, 2, 3, 0, 0, 1], [4, 1, 2, 0, 0, 0], [2, 3, 3, 1, 0, 0],
    [3, 4, 1, 0, 1, 0], [4, 2, 2, 0, 0, 0], [2, 3, 4, 0, 0, 0],
    [3, 2, 2, 0, 0, 0], [4, 3, 1, 1, 0, 0], [2, 4, 3, 0, 0, 0],
    [3, 2, 2, 0, 0, 1], [4, 1, 3, 0, 0, 0], [2, 3, 2, 0, 0, 0],
    [3, 4, 1, 1, 0, 0], [4, 2, 3, 0, 0, 0], [2, 3, 2, 0, 1, 0],
    [3, 2, 4, 0, 0, 0], [4, 3, 2, 0, 0, 0], [3, 3, 2, 0, 0, 0],
    [2, 4, 1, 0, 0, 0], [4, 2, 3, 1, 0, 0], [3, 1, 4, 0, 0, 0],
    [2, 3, 3, 0, 0, 0]
])

X_multi = np.vstack([healthy_reports, failing_reports])
y_multi = np.array([0]*25 + [1]*25)
word_names = ['smoke', 'noise', 'leak', 'normal', 'smooth', 'clean']

X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(
    X_multi, y_multi, test_size=0.25, random_state=42
)

mnb = MultinomialNB()
mnb.fit(X_m_train, y_m_train)
m_acc = accuracy_score(y_m_test, mnb.predict(X_m_test))

# mnb.feature_log_prob_ stores log(P(word | class)) for each word/class
# np.exp converts from log-probability back to regular probability
word_probs = np.exp(mnb.feature_log_prob_)

print(f"  Training reports: {len(X_m_train)}  |  Test accuracy: {m_acc:.2%}")
print()
print("  Learned word probabilities per class:")
print(f"  {'Word':<10} {'P(word|Healthy)':>17} {'P(word|Failing)':>17}")
print("  " + "-" * 47)
for i, word in enumerate(word_names):
    marker = ' ← key discriminator' if abs(word_probs[0][i] - word_probs[1][i]) > 0.15 else ''
    print(f"  {word:<10} {word_probs[0][i]:>17.3f} {word_probs[1][i]:>17.3f}{marker}")

# New report: heavy smoke + noise → should predict Failing
new_report = np.array([[4, 3, 2, 0, 0, 0]])
prob_m = mnb.predict_proba(new_report)[0]
pred_m = mnb.predict(new_report)[0]
print(f"\n  New report (smoke=4, noise=3, leak=2, normal=0, smooth=0, clean=0):")
print(f"  → P(Healthy): {prob_m[0]:.2%}  |  P(Failing): {prob_m[1]:.2%}")
print(f"  → Prediction: {'Failing' if pred_m == 1 else 'Healthy'}")
print("=" * 65)


# =============================================================
# VARIANT 3: BernoulliNB — Binary Inspection Checklist
# =============================================================
print()
print("=" * 65)
print("  VARIANT 3: BernoulliNB — Binary Inspection Checklist")
print("=" * 65)
print("  Data type : Binary only — 1 (PASS) or 0 (FAIL)")
print("  Assumption: Each feature is a coin flip per class")
print("  Checks    : [oil_ok, brakes_ok, tires_ok, coolant_ok, belt_ok]")
print()

# Healthy trucks: pass most checks (high pass rate ~88%)
# Failing trucks: fail most checks (low pass rate ~28%)
healthy_checks = (np.random.rand(60, 5) > 0.12).astype(int)
failing_checks = (np.random.rand(60, 5) > 0.72).astype(int)

X_bern = np.vstack([healthy_checks, failing_checks])
y_bern = np.array([0]*60 + [1]*60)
check_names = ['oil_ok', 'brakes_ok', 'tires_ok', 'coolant_ok', 'belt_ok']

X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(
    X_bern, y_bern, test_size=0.2, random_state=42
)

bnb = BernoulliNB()
bnb.fit(X_b_train, y_b_train)
b_acc = accuracy_score(y_b_test, bnb.predict(X_b_test))

# bnb.feature_log_prob_ = log P(feature=1 | class)
# np.exp converts back to probability
check_probs = np.exp(bnb.feature_log_prob_)

print(f"  Training inspections: {len(X_b_train)}  |  Test accuracy: {b_acc:.2%}")
print()
print("  Probability each check PASSES (feature=1) per class:")
print(f"  {'Check':<14} {'P(PASS|Healthy)':>16} {'P(PASS|Failing)':>16}")
print("  " + "-" * 48)
for i, check in enumerate(check_names):
    print(f"  {check:<14} {check_probs[0][i]:>16.3f} {check_probs[1][i]:>16.3f}")

# Truck with oil FAIL, brakes FAIL, rest OK
new_inspection = np.array([[0, 0, 1, 1, 0]])
prob_b = bnb.predict_proba(new_inspection)[0]
pred_b = bnb.predict(new_inspection)[0]
print(f"\n  New truck (oil=FAIL, brakes=FAIL, tires=OK, coolant=OK, belt=FAIL):")
print(f"  → P(Healthy): {prob_b[0]:.2%}  |  P(Failing): {prob_b[1]:.2%}")
print(f"  → Prediction: {'Failing' if pred_b == 1 else 'Healthy'}")
print("=" * 65)


# ── VISUALIZATION ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
plt.subplots_adjust(wspace=0.30)

# LEFT: GaussianNB — 2D decision boundary showing bell-curve-based boundary
x_min = X_gauss[:, 0].min() - 5;  x_max = X_gauss[:, 0].max() + 5
y_min = X_gauss[:, 1].min() - 0.05; y_max = X_gauss[:, 1].max() + 0.05
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),
                     np.arange(y_min, y_max, 0.008))
Z_g = gnb.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
axes[0].contourf(xx, yy, Z_g, alpha=0.25, cmap='bwr')
axes[0].scatter(X_gauss[y_gauss==0, 0], X_gauss[y_gauss==0, 1],
                color='blue', edgecolor='k', s=50, alpha=0.7, label='Healthy (0)')
axes[0].scatter(X_gauss[y_gauss==1, 0], X_gauss[y_gauss==1, 1],
                color='red',  edgecolor='k', s=50, alpha=0.7, label='Failing (1)')
axes[0].scatter(new_truck_g[0][0], new_truck_g[0][1], color='green',
                marker='*', s=320, zorder=6,
                label=f'New truck → {"Failing" if pred_g==1 else "Healthy"}')
axes[0].set_xlabel('Speed (km/h)')
axes[0].set_ylabel('Vibration (G-force)')
axes[0].set_title(f'GaussianNB — Continuous Sensor Data\n'
                  f'Accuracy: {g_acc:.1%}  |  100 trucks\n'
                  f'Boundary formed where bell curves from each class intersect')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.2)

# MIDDLE: MultinomialNB — word probability bar chart per class
x_pos = np.arange(len(word_names))
width = 0.35
axes[1].bar(x_pos - width/2, word_probs[0], width,
            label='P(word | Healthy)', color='steelblue', edgecolor='k', alpha=0.85)
axes[1].bar(x_pos + width/2, word_probs[1], width,
            label='P(word | Failing)', color='tomato', edgecolor='k', alpha=0.85)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(word_names, rotation=25, ha='right', fontsize=10)
axes[1].set_ylabel('P(word | class)')
axes[1].set_title(f'MultinomialNB — Maintenance Report Word Counts\n'
                  f'Accuracy: {m_acc:.1%}  |  50 reports\n'
                  f'"smoke/noise/leak" → Failing  |  "normal/smooth/clean" → Healthy')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.2, axis='y')
axes[1].set_ylim(0, max(word_probs.max(), 0.55) + 0.05)

# RIGHT: BernoulliNB — check pass probability per class
x_pos_b = np.arange(len(check_names))
axes[2].bar(x_pos_b - width/2, check_probs[0], width,
            label='P(PASS | Healthy)', color='steelblue', edgecolor='k', alpha=0.85)
axes[2].bar(x_pos_b + width/2, check_probs[1], width,
            label='P(PASS | Failing)', color='tomato', edgecolor='k', alpha=0.85)
axes[2].set_xticks(x_pos_b)
axes[2].set_xticklabels(check_names, rotation=25, ha='right', fontsize=10)
axes[2].set_ylabel('Probability check passes (=1)')
axes[2].set_title(f'BernoulliNB — Binary Inspection Checklist\n'
                  f'Accuracy: {b_acc:.1%}  |  120 inspections\n'
                  f'Healthy trucks pass most checks  |  Failing trucks fail many')
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.2, axis='y')
axes[2].set_ylim(0, 1.15)

# Add value labels on top of BernoulliNB bars
for i in range(len(check_names)):
    axes[2].text(i - width/2, check_probs[0][i] + 0.02, f'{check_probs[0][i]:.2f}',
                 ha='center', fontsize=7.5, color='steelblue', fontweight='bold')
    axes[2].text(i + width/2, check_probs[1][i] + 0.02, f'{check_probs[1][i]:.2f}',
                 ha='center', fontsize=7.5, color='tomato', fontweight='bold')

plt.suptitle('NAIVE BAYES — 3 Variants: Same Bayes Theorem, Different Data Assumptions',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
