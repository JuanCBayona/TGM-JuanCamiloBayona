import json
import tempfile

from pathlib import Path

import numpy as np
import pandas as pd

from generative_metrics import GenerativeMetrics
from report import ReportGenerator

rng = np.random.default_rng(0)

# ---- FID sanity ----------------------------------------------------

d = 64
n = 4000

a = rng.normal(0.0, 1.0, size=(n, d))
b = rng.normal(0.0, 1.0, size=(n, d))

fid_same = GenerativeMetrics.frechet_inception_distance(a, a)
print(f"FID(a, a)            = {fid_same:.6f}  (expect ~0)")
assert fid_same < 1e-6

fid_iid = GenerativeMetrics.frechet_inception_distance(a, b)
print(f"FID(a, b) iid        = {fid_iid:.4f}  (expect small, sampling noise)")
assert fid_iid < 5.0

# mean-shifted: FID should equal ||mu1 - mu2||^2 when covs match
shift = np.zeros(d)
shift[0] = 3.0
c = rng.normal(0.0, 1.0, size=(n, d)) + shift

fid_shift = GenerativeMetrics.frechet_inception_distance(a, c)
print(f"FID(a, a+3e0)        = {fid_shift:.4f}  (expect ~9 + noise)")
assert abs(fid_shift - 9.0) < 1.5

# symmetry
fid_ab = GenerativeMetrics.frechet_inception_distance(a, c)
fid_ba = GenerativeMetrics.frechet_inception_distance(c, a)
print(f"symmetry             = {abs(fid_ab - fid_ba):.3e}  (expect ~0)")
assert abs(fid_ab - fid_ba) < 1e-6

# scale-shifted covariance
e = rng.normal(0.0, 2.0, size=(n, d))
fid_cov = GenerativeMetrics.frechet_inception_distance(a, e)
print(f"FID(N(0,1), N(0,4))  = {fid_cov:.4f}  (expect ~d*(1+4-2*2)=64)")
assert abs(fid_cov - 64.0) < 8.0

# degenerate / rank-deficient input must not crash or go negative
few = rng.normal(size=(5, d))
fid_few = GenerativeMetrics.frechet_inception_distance(few, a)
print(f"FID(5 samples, a)    = {fid_few:.4f}  (must be finite, >= 0)")
assert np.isfinite(fid_few) and fid_few >= 0.0

# ---- Inception Score sanity ----------------------------------------

k = 10
m = 1000

# perfectly confident + perfectly balanced -> IS == number of classes
onehot = np.zeros((m, k))
onehot[np.arange(m), np.arange(m) % k] = 1.0

is_mean, is_std = GenerativeMetrics.inception_score(onehot, splits=10)
print(f"IS(balanced one-hot) = {is_mean:.4f} +/- {is_std:.4f}  (expect slightly under {k})")
assert 0.9 * k <= is_mean <= k + 1e-6

# with a single split the marginal is exactly uniform -> IS == k exactly
is_exact, _ = GenerativeMetrics.inception_score(onehot, splits=1)
print(f"IS(one-hot, 1 split) = {is_exact:.6f}  (expect exactly {k})")
assert abs(is_exact - k) < 1e-9

# every image gets the same distribution -> no diversity -> IS == 1
flat = np.tile(rng.dirichlet(np.ones(k)), (m, 1))
is_flat, _ = GenerativeMetrics.inception_score(flat, splits=10)
print(f"IS(identical probs)  = {is_flat:.4f}  (expect ~1)")
assert abs(is_flat - 1.0) < 1e-3

# uniform predictions -> IS == 1
uniform = np.full((m, k), 1.0 / k)
is_uniform, _ = GenerativeMetrics.inception_score(uniform, splits=10)
print(f"IS(uniform probs)    = {is_uniform:.4f}  (expect ~1)")
assert abs(is_uniform - 1.0) < 1e-3

# IS is bounded by the number of classes
real = rng.dirichlet(np.ones(k) * 0.1, size=m)
is_real, _ = GenerativeMetrics.inception_score(real, splits=10)
print(f"IS(dirichlet)        = {is_real:.4f}  (expect 1 <= IS <= {k})")
assert 1.0 <= is_real <= k + 1e-6

# fewer images than splits must not crash
is_tiny, _ = GenerativeMetrics.inception_score(onehot[:3], splits=10)
print(f"IS(3 images)         = {is_tiny:.4f}  (must be finite)")
assert np.isfinite(is_tiny)

# determinism (fixed seed shuffle)
r1 = GenerativeMetrics.inception_score(real)
r2 = GenerativeMetrics.inception_score(real)
assert r1 == r2
print("IS is deterministic  = OK")

# ---- evaluate() + downstream json/csv/report ------------------------

gen = GenerativeMetrics.evaluate(
    source_features=a,
    objective_features=c,
    source_probabilities=real,
    objective_probabilities=onehot,
    source_label="source",
    objective_label="objective"
)

print("\nGenerativeMetrics.evaluate keys:")
for key, value in gen.items():
    print(f"  {key:<25}: {value:.6f}")

assert set(gen) == {
    "FID",
    "IS_source_mean",
    "IS_source_std",
    "IS_objective_mean",
    "IS_objective_std"
}
assert all(isinstance(v, float) for v in gen.values())

# simulate a full metrics dict like run.py builds it
embedding = {
    "KL": 0.4, "JS": 0.1, "EMD": 0.02, "MMD": 0.3,
    "Frechet": 12.0, "KS": 0.25, "NLL": 55.0,
    "CosineSimilarity": 0.81
}
merged = dict(embedding)
merged.update(gen)

# print_results-style formatting must not blow up
for key, value in merged.items():
    _ = f"{key:<25}: {value:.6f}"

# json + csv writers used by save_embedding_metrics
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    (tmp / "source_vs_objective").mkdir()
    (tmp / "source_vs_originals").mkdir()

    with open(tmp / "source_vs_objective" / "metrics.json", "w") as f:
        json.dump(merged, f, indent=4)
    pd.DataFrame([merged]).to_csv(
        tmp / "source_vs_objective" / "metrics.csv", index=False
    )

    worse = dict(merged)
    worse["FID"] = merged["FID"] * 2.0
    worse["KL"] = 0.8
    worse["IS_originals_mean"] = worse.pop("IS_objective_mean")
    worse["IS_originals_std"] = worse.pop("IS_objective_std")

    with open(tmp / "source_vs_originals" / "metrics.json", "w") as f:
        json.dump(worse, f, indent=4)

    ReportGenerator.create_report(
        output_file=tmp / "report.md",
        source_path="/data/source",
        objective_path="/data/objective",
        originals_path="/data/originals",
        embedding_metrics=merged,
        conditional_metrics={"MSE_mean": 12.0, "PSNR_mean": 30.0, "SSIM_mean": 0.9}
    )

    report_text = (tmp / "report.md").read_text()
    assert "FID" in report_text
    assert "IS_source_mean" in report_text
    assert "IS_objective_mean" in report_text
    assert "FID: 50.00% improvement" in report_text
    csv_cols = pd.read_csv(
        tmp / "source_vs_objective" / "metrics.csv"
    ).columns.tolist()
    assert "FID" in csv_cols and "IS_source_mean" in csv_cols

    print("\n--- report.md (excerpt) ---")
    print("\n".join(report_text.splitlines()[:35]))

print("\nALL TESTS PASSED")
