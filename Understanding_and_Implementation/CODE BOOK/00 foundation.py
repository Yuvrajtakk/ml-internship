# ==============================================================
# 00_foundation.py
# ML FROM SCRATCH — The Core Loop Inside Every Algorithm
# ==============================================================
# No sklearn here. Pure Python only.
#
# WHAT THIS FILE ANSWERS:
#   "What is the machine actually doing when it learns?"
#
# ANSWER: Make a prediction → measure the error → nudge the
#         weights to reduce that error → repeat until error ≈ 0
#
# This exact loop runs inside sklearn's .fit() every time
# you call it. The algorithms differ in HOW they calculate
# the update — not WHETHER they do this loop.
# ==============================================================


# ── SECTION 1: One Point, One Weight ──────────────────────────
# The absolute minimum version of ML: one input, one output,
# one parameter (w) to learn.

print("=" * 55)
print("  SECTION 1: One data point, one weight")
print("=" * 55)

input_x  = 2      # Feature: e.g., distance driven (km)
target_y = 6      # Label: e.g., fuel cost we want to predict
w        = 1.0    # Weight: the model's only parameter (starts as a guess)

prediction = input_x * w            # 2 × 1.0 = 2.0 — wrong
error      = target_y - prediction  # 6 - 2.0 = 4.0 — we're off by 4

print(f"  Weight (w)      : {w}")
print(f"  Prediction      : {prediction}  (input × w = 2 × 1.0)")
print(f"  Target          : {target_y}")
print(f"  Error           : {error}  ← positive means w should go UP")


# ── SECTION 2: Learning Rate + Epochs ─────────────────────────
# We can't just jump to the perfect weight in one step.
# learning_rate controls how big a step we take.
# Too large → oscillates and diverges. Too small → takes forever.
# epoch = one full pass over the data.

print("\n" + "=" * 55)
print("  SECTION 2: Introducing learning rate and epochs")
print("=" * 55)

w             = 1.0
learning_rate = 0.05   # Tested this value — stable and converges

for epoch in range(10):
    prediction = input_x * w
    error      = target_y - prediction
    # Update rule: step in the direction that reduces error.
    # Multiplying by input_x scales the step by the input magnitude.
    w = w + (learning_rate * error * input_x)
    print(f"  Epoch {epoch+1:2d} | Pred: {prediction:.4f} | Error: {error:.4f} | w: {w:.4f}")

print(f"\n  Target w was 3.0 (because 2 × 3 = 6). Learned: {w:.4f}")


# ── SECTION 3: Multiple Data Points (SGD) ─────────────────────
# Real datasets have many points, not just one.
# We loop through ALL points, updating w after each one.
# This is called Stochastic Gradient Descent (SGD).

print("\n" + "=" * 55)
print("  SECTION 3: Multiple data points — Stochastic GD")
print("=" * 55)

inputs  = [1, 2, 3, 4, 5, 6]
targets = [3, 6, 9, 12, 15, 18]   # Rule: target = 3 × input

w             = 1.0
learning_rate = 0.02

for epoch in range(6):
    total_error = 0
    for x, y in zip(inputs, targets):   # zip pairs input[i] with target[i]
        prediction  = x * w
        error       = y - prediction
        w           = w + (learning_rate * error * x)
        total_error += abs(error)
    avg_error = total_error / len(inputs)
    print(f"  Epoch {epoch+1} | Avg error: {avg_error:.5f} | w learned so far: {w:.5f}")

print(f"\n  True w = 3.0 | Learned w = {w:.5f}")


# ── SECTION 4: Bias — The Intercept Term ──────────────────────
# Without bias, the line must pass through the origin (0,0).
# Adding b allows the line to shift up or down freely.
# prediction = (w × x) + b

print("\n" + "=" * 55)
print("  SECTION 4: Adding bias (intercept)")
print("=" * 55)
print("  Rule to learn: target = 2×x + 3  (w=2, b=3)")
print()

inputs  = [1, 2, 3, 4, 5, 6]
targets = [5, 7, 9, 11, 13, 15]   # 2×1+3=5, 2×2+3=7, etc.

w             = 1.0
b             = 0.0
learning_rate = 0.02

for epoch in range(200):
    for x, y in zip(inputs, targets):
        prediction = (x * w) + b
        error      = y - prediction
        w = w + (learning_rate * error * x)   # Weight update
        b = b + (learning_rate * error * 1)   # Bias update (input for bias is always 1)
    if (epoch + 1) % 50 == 0:
        last_pred  = (inputs[-1] * w) + b
        last_error = targets[-1] - last_pred
        print(f"  Epoch {epoch+1:3d} | w: {w:.5f} | b: {b:.5f} | last error: {last_error:.6f}")

print(f"\n  True    : w=2.0, b=3.0")
print(f"  Learned : w={w:.5f}, b={b:.5f}")


# ── SECTION 5: Visualization ───────────────────────────────────
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Train again, tracking error per epoch for the left plot
w_vis, b_vis = 1.0, 0.0
lr_vis = 0.02
errors_over_time = []

for epoch in range(100):
    mse = 0
    for x, y in zip(inputs, targets):
        pred  = (x * w_vis) + b_vis
        err   = y - pred
        w_vis = w_vis + (lr_vis * err * x)
        b_vis = b_vis + (lr_vis * err)
        mse  += err ** 2
    errors_over_time.append(mse / len(inputs))

# LEFT: Error curve — should drop steeply then flatten
ax1.plot(errors_over_time, color='tomato', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Mean Squared Error')
ax1.set_title('Error Drops as the Machine Learns\n(flat at the bottom = converged)')
ax1.grid(True, alpha=0.3)
ax1.annotate(f'Final MSE ≈ {errors_over_time[-1]:.5f}',xy=(95, errors_over_time[-1]),xytext=(60, errors_over_time[3]),arrowprops=dict(arrowstyle='->', color='black'),fontsize=9)

# RIGHT: Learned line vs actual data points
predictions_vis = [(x * w_vis) + b_vis for x in inputs]
ax2.scatter(inputs, targets, color='steelblue', s=120, zorder=5, label='Actual data')
ax2.plot(inputs, predictions_vis, color='tomato', linewidth=2.5,label=f'Learned: {w_vis:.2f}×x + {b_vis:.2f}')
ax2.set_xlabel('Input (x)')
ax2.set_ylabel('Output (y)')
ax2.set_title('Learned Line vs Actual Data\n(dots should fall on the line)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle('ML FROM SCRATCH — Weight + Bias + Error Reduction', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()