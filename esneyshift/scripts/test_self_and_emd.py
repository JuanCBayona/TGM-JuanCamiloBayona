import numpy as np

from scipy.stats import wasserstein_distance

from metrics import Metrics
from generative_metrics import GenerativeMetrics
from validation import is_self_comparison

rng = np.random.default_rng(0)

# =====================================================================
# 1. SELF-COMPARISON -> everything must collapse to 0 (cosine -> 1)
# =====================================================================

x = rng.normal(size=(3000, 128)) @ rng.normal(size=(128, 256))

print("=" * 62)
print("SELF-COMPARISON  Metrics.evaluate(X, X)")
print("=" * 62)

self_metrics = Metrics.evaluate(x, x)

print()
for key, value in self_metrics.items():
    print(f"  {key:<20}: {value:.10f}")

assert abs(self_metrics["KL"]) < 1e-9
assert abs(self_metrics["JS"]) < 1e-9
assert abs(self_metrics["EMD"]) < 1e-9
assert abs(self_metrics["MMD"]) < 1e-9
assert abs(self_metrics["KS"]) < 1e-9
assert abs(self_metrics["Frechet"]) < 1e-4
assert abs(self_metrics["CosineSimilarity"] - 1.0) < 1e-9
# NLL is a cross-entropy, not a divergence -> finite, not zero
assert np.isfinite(self_metrics["NLL"])
print("\n  -> distances ~0, cosine ~1.0, NLL finite (expected)  OK")

# FID on identical inception activations
act = rng.normal(size=(3000, 2048))
fid_self = GenerativeMetrics.frechet_inception_distance(act, act)
print(f"\n  FID(X, X)           : {fid_self:.10f}   (expect ~0)")
assert fid_self < 1e-3

# IS must be identical for both sides
probs = rng.dirichlet(np.ones(1000) * 0.05, size=500)
is_a = GenerativeMetrics.inception_score(probs)
is_b = GenerativeMetrics.inception_score(probs)
assert is_a == is_b
print(f"  IS(X) == IS(X)      : {is_a[0]:.6f}  (identical)  OK")

assert is_self_comparison("/tmp", "/tmp/") is True
assert is_self_comparison("/tmp", "/home") is False
print("  is_self_comparison  : OK")

# =====================================================================
# 2. EMD FIX -> old version was blind to a pure location shift
# =====================================================================

print()
print("=" * 62)
print("EMD FIX")
print("=" * 62)


def old_emd(source_reduced, objective_reduced):
    """The previous implementation: Wasserstein over BIN MASSES."""
    s_h, o_h = Metrics.compute_histograms(source_reduced, objective_reduced)
    return float(np.mean([
        wasserstein_distance(h1, h2) for h1, h2 in zip(s_h, o_h)
    ]))


a = rng.normal(0.0, 1.0, size=(4000, 20))

for shift in [0.0, 0.5, 2.0, 10.0]:

    b = rng.normal(0.0, 1.0, size=(4000, 20)) + shift

    old = old_emd(a, b)
    new = Metrics.emd_distance(a, b)

    print(
        f"  mean shift = {shift:5.1f}  ->  "
        f"old EMD = {old:.6f}   "
        f"new EMD = {new:.6f}"
    )

# the true W1 between N(0,1) and N(shift,1) is exactly `shift`
b10 = rng.normal(0.0, 1.0, size=(20000, 20)) + 10.0
a10 = rng.normal(0.0, 1.0, size=(20000, 20))
new10 = Metrics.emd_distance(a10, b10)
old10 = old_emd(a10, b10)

print()
print("  ground truth W1 for a shift of 10.0 = 10.0")
print(f"  new EMD recovers                    = {new10:.4f}   <- correct")
print(f"  old EMD reported                    = {old10:.4f}   <- blind")

assert abs(new10 - 10.0) < 0.1, "fixed EMD must recover the true W1"
assert old10 < 0.05, "old EMD stayed ~0 despite a shift of 10 sigma"

# and it still returns exactly 0 for identical inputs
assert Metrics.emd_distance(a, a) == 0.0
print("\n  EMD(X, X) == 0.0  OK")

# monotonicity: EMD must grow with the shift
emds = [
    Metrics.emd_distance(a, rng.normal(size=(4000, 20)) + s)
    for s in [0.0, 1.0, 2.0, 4.0]
]
assert all(emds[i] < emds[i + 1] for i in range(len(emds) - 1))
print("  EMD is monotonic in the shift  OK")

print()
print("ALL TESTS PASSED")
