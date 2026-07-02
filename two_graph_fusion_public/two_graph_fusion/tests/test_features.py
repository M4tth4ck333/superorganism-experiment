"""Unit tests for the five behavioral features.

These tests intentionally avoid touching real TwiBot-22 data; they generate
synthetic event timelines whose behavior under each feature is analytically
known.

Run with::

    python -m pytest two_graph_fusion/tests/

or, without pytest::

    python -m unittest two_graph_fusion.tests.test_features
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from two_graph_fusion.features.activity_duration import activity_duration
from two_graph_fusion.features.analysis import feature_separation
from two_graph_fusion.features.burstiness import burstiness
from two_graph_fusion.features.calibration import (
    calibrate_circadian_humanlike,
    refit_d_shape_with_reference,
)
from two_graph_fusion.features.circadian import (
    CircadianConfig,
    circadian_zscore,
    humanlike,
)
from two_graph_fusion.features.memory import memory_coefficient
from two_graph_fusion.features.pipeline import (
    PipelineConfig,
    compute_behavioral_features,
)
from two_graph_fusion.features.shape_divergence import (
    fit_reference_bins,
    shape_divergence,
)


class TestBurstiness(unittest.TestCase):
    def test_periodic_is_minus_one(self) -> None:
        iats = np.full(100, 60.0)
        self.assertAlmostEqual(burstiness(iats), -1.0, places=6)

    def test_poisson_is_near_zero(self) -> None:
        rng = np.random.default_rng(0)
        iats = rng.exponential(scale=60.0, size=20_000)
        self.assertLess(abs(burstiness(iats)), 0.05)

    def test_extreme_burst_is_near_plus_one(self) -> None:
        # With n=1 outlier among (n-1) zeros, the algebraic cap of B is
        # (sqrt(n-1) - 1) / (sqrt(n-1) + 1); we need n >= ~10000 for B > 0.95.
        iats = np.array([1e-6] * 9999 + [1e6])
        self.assertGreater(burstiness(iats), 0.95)

    def test_too_few_samples_returns_nan(self) -> None:
        self.assertTrue(math.isnan(burstiness([5.0])))

    def test_negative_iats_raise(self) -> None:
        with self.assertRaises(ValueError):
            burstiness([1.0, -2.0, 3.0])


class TestMemory(unittest.TestCase):
    def test_constant_iats_return_nan(self) -> None:
        self.assertTrue(math.isnan(memory_coefficient(np.full(50, 60.0))))

    def test_poisson_is_near_zero(self) -> None:
        rng = np.random.default_rng(1)
        iats = rng.exponential(scale=60.0, size=20_000)
        self.assertLess(abs(memory_coefficient(iats)), 0.05)

    def test_positive_memory(self) -> None:
        rng = np.random.default_rng(2)
        n = 5000
        x = rng.normal(0.0, 1.0, size=n)
        iats = np.empty(n)
        iats[0] = x[0]
        rho = 0.7
        for i in range(1, n):
            iats[i] = rho * iats[i - 1] + math.sqrt(1 - rho * rho) * x[i]
        iats = iats - iats.min() + 1.0
        m = memory_coefficient(iats)
        self.assertGreater(m, 0.5)


class TestActivityDuration(unittest.TestCase):
    def test_full_span(self) -> None:
        t = np.linspace(0.0, 100.0, 50)
        self.assertAlmostEqual(activity_duration(t, 0.0, 100.0), 1.0, places=6)

    def test_half_span(self) -> None:
        t = np.linspace(0.0, 50.0, 50)
        self.assertAlmostEqual(activity_duration(t, 0.0, 100.0), 0.5, places=6)

    def test_empty(self) -> None:
        self.assertTrue(math.isnan(activity_duration([], 0.0, 100.0)))

    def test_clip_above_one(self) -> None:
        t = np.array([0.0, 200.0])
        self.assertEqual(activity_duration(t, 0.0, 100.0), 1.0)


class TestCircadian(unittest.TestCase):
    def test_concentrated_in_one_hour_high_z(self) -> None:
        # All events at hour 12 of consecutive days
        n_days = 30
        timestamps = np.array(
            [d * 86400 + 12 * 3600 for d in range(n_days) for _ in range(5)]
        )
        cfg = CircadianConfig(n_perm=200, seed=0)
        z = circadian_zscore(timestamps, cfg)
        self.assertGreater(z, 5.0)

    def test_uniform_low_z(self) -> None:
        rng = np.random.default_rng(3)
        timestamps = rng.uniform(0.0, 30 * 86400, size=5000)
        cfg = CircadianConfig(n_perm=200, seed=1)
        z = circadian_zscore(timestamps, cfg)
        self.assertLess(abs(z), 4.0)

    def test_humanlike_sigmoid_default_monotone(self) -> None:
        cfg = CircadianConfig(target_z=2.0, sigma_z=1.0)
        # 0.5 at target_z, monotone increasing in z.
        self.assertAlmostEqual(humanlike(2.0, cfg), 0.5, places=6)
        self.assertGreater(humanlike(10.0, cfg), humanlike(2.0, cfg))
        self.assertLess(humanlike(-10.0, cfg), humanlike(2.0, cfg))
        # Bounded in [0, 1] even at extreme values (saturation is allowed).
        self.assertGreaterEqual(humanlike(-1e6, cfg), 0.0)
        self.assertLessEqual(humanlike(1e6, cfg), 1.0)

    def test_humanlike_gaussian_peaks_at_target(self) -> None:
        cfg = CircadianConfig(target_z=2.0, sigma_z=1.0, mapping="gaussian")
        self.assertAlmostEqual(humanlike(2.0, cfg), 1.0, places=6)
        self.assertLess(humanlike(7.0, cfg), 1e-5)


class TestShapeDivergence(unittest.TestCase):
    def test_identical_distribution_close_to_one(self) -> None:
        rng = np.random.default_rng(4)
        ref = rng.exponential(scale=300.0, size=20_000)
        bins = fit_reference_bins(ref)
        user_iats = rng.exponential(scale=300.0, size=2000)
        d = shape_divergence(user_iats, bins)
        self.assertGreater(d, 0.95)

    def test_very_different_distribution_low_score(self) -> None:
        rng = np.random.default_rng(5)
        ref = rng.exponential(scale=300.0, size=20_000)
        bins = fit_reference_bins(ref)
        # Constant-rate bot: every IAT is exactly 60s
        bot_iats = np.full(500, 60.0)
        d = shape_divergence(bot_iats, bins)
        self.assertLess(d, 0.6)

    def test_too_few_iats_returns_nan(self) -> None:
        rng = np.random.default_rng(6)
        ref = rng.exponential(scale=300.0, size=1000)
        bins = fit_reference_bins(ref)
        self.assertTrue(math.isnan(shape_divergence([1.0, 2.0], bins)))


class TestPipeline(unittest.TestCase):
    def test_pipeline_honest_reference_shrinks_bot_dshape(self) -> None:
        """D_shape using honest-only reference should be lower for bots
        than D_shape using the pooled reference."""
        rng = np.random.default_rng(13)
        users: dict[str, np.ndarray] = {}
        for i in range(8):
            iats = rng.exponential(scale=600.0, size=300)
            users[f"h{i}"] = np.cumsum(iats)
        for i in range(8):
            users[f"b{i}"] = np.arange(60.0, 60.0 * 301, 60.0)
        all_ts = np.concatenate(list(users.values()))
        window = (float(all_ts.min()), float(all_ts.max()))

        from two_graph_fusion.features import PipelineConfig

        pooled = compute_behavioral_features(
            users,
            window=window,
            config=PipelineConfig(
                n_min_events=50,
                circadian=CircadianConfig(n_perm=10, seed=0),
            ),
        )
        honest_only = compute_behavioral_features(
            users,
            window=window,
            config=PipelineConfig(
                n_min_events=50,
                circadian=CircadianConfig(n_perm=10, seed=0),
                reference_user_ids=frozenset(f"h{i}" for i in range(8)),
            ),
        )
        bots = [f"b{i}" for i in range(8)]
        # Honest reference should make bots look less like the reference.
        self.assertLess(
            honest_only.raw.loc[bots, "D_shape"].mean(),
            pooled.raw.loc[bots, "D_shape"].mean(),
        )

    def test_pipeline_basic(self) -> None:
        rng = np.random.default_rng(7)
        users = {}
        # 5 "human-like" users: bursty Poisson over a long window
        for i in range(5):
            iats = rng.exponential(scale=600.0, size=300)
            ts = np.cumsum(iats)
            ts += rng.uniform(0.0, 86400)
            users[f"u_human_{i}"] = ts
        # 5 "bot-like" users: constant cadence
        for i in range(5):
            ts = np.arange(60, 60 * 301, 60.0)
            ts += rng.uniform(0.0, 86400)
            users[f"u_bot_{i}"] = ts

        all_ts = np.concatenate(list(users.values()))
        window = (float(all_ts.min()), float(all_ts.max()))
        cfg = PipelineConfig(
            n_min_events=50,
            circadian=CircadianConfig(n_perm=20, seed=0),
        )
        result = compute_behavioral_features(users, window=window, config=cfg)

        # All 10 users qualify
        self.assertEqual(result.raw.dropna(subset=["B_G"]).shape[0], 10)
        # Bots have lower (more negative) B_G than humans (constant cadence)
        bot_b = result.raw.loc[[f"u_bot_{i}" for i in range(5)], "B_G"]
        human_b = result.raw.loc[[f"u_human_{i}" for i in range(5)], "B_G"]
        self.assertLess(bot_b.mean(), human_b.mean())
        # Z-scored frame has the right columns and same index
        self.assertEqual(len(result.zscored), len(result.raw))


class TestSeparation(unittest.TestCase):
    def test_perfect_separation_auc_one(self) -> None:
        import pandas as pd

        df = pd.DataFrame(
            {
                "f": list(range(10)) + list(range(100, 110)),
                "label": ["neg"] * 10 + ["pos"] * 10,
            }
        )
        sep = feature_separation(df, "f", "label", "pos")
        self.assertAlmostEqual(sep.auc, 1.0, places=6)
        self.assertEqual(sep.auc_direction, "+")

    def test_inverse_separation_auc_one(self) -> None:
        import pandas as pd

        df = pd.DataFrame(
            {
                "f": list(range(100, 110)) + list(range(10)),
                "label": ["neg"] * 10 + ["pos"] * 10,
            }
        )
        sep = feature_separation(df, "f", "label", "pos")
        self.assertAlmostEqual(sep.auc, 1.0, places=6)
        self.assertEqual(sep.auc_direction, "-")

    def test_no_separation_auc_half(self) -> None:
        import pandas as pd

        rng = np.random.default_rng(0)
        f = rng.normal(size=2000)
        labels = rng.choice(["pos", "neg"], size=2000)
        df = pd.DataFrame({"f": f, "label": labels})
        sep = feature_separation(df, "f", "label", "pos")
        self.assertAlmostEqual(sep.auc, 0.5, delta=0.04)


class TestCalibration(unittest.TestCase):
    def test_refit_d_shape_runs(self) -> None:
        import pandas as pd

        rng = np.random.default_rng(11)
        users = {
            f"h{i}": np.cumsum(rng.exponential(600.0, size=200))
            for i in range(5)
        }
        users.update(
            {f"b{i}": np.cumsum(np.full(200, 60.0)) for i in range(5)}
        )
        users = {uid: np.unique(ts) for uid, ts in users.items()}
        d, bins = refit_d_shape_with_reference(
            users, reference_user_ids=[f"h{i}" for i in range(5)], cap=None
        )
        humans = pd.Series({uid: d[uid] for uid in users if uid.startswith("h")})
        bots = pd.Series({uid: d[uid] for uid in users if uid.startswith("b")})
        self.assertGreater(humans.mean(), bots.mean())
        self.assertGreaterEqual(bins.edges.size, 9)

    def test_calibrate_circadian_humanlike_sigmoid_default(self) -> None:
        import pandas as pd

        # Honest users below z=15, bots above. The sigmoid default puts the
        # 0.5 crossover at honest p25 = 10, so honest median (z=11) is
        # slightly above 0.5 and bots (z=50+) are near 1.0.
        raw = pd.DataFrame(
            {
                "C_24": [10.0, 12.0, 11.0, 9.0, 13.0, 50.0, 60.0, 55.0],
                "label": ["human"] * 5 + ["bot"] * 3,
            },
            index=[f"u{i}" for i in range(8)],
        )
        humanlike_series, cfg = calibrate_circadian_humanlike(
            raw, label_col="label", honest_label="human", min_honest=5
        )
        # cfg.target_z is honest p25
        self.assertAlmostEqual(cfg.target_z, 10.0, places=6)
        # Sigmoid is monotone; high-z bot users score higher than low-z humans.
        self.assertGreater(humanlike_series.loc["u5"], humanlike_series.loc["u3"])
        # u3 has z=9 (below crossover) so sigmoid < 0.5; u2 has z=11 so >= 0.5.
        self.assertLess(humanlike_series.loc["u3"], 0.5)
        self.assertGreater(humanlike_series.loc["u2"], 0.5)

    def test_calibrate_circadian_humanlike_gaussian_form(self) -> None:
        import pandas as pd

        raw = pd.DataFrame(
            {
                "C_24": [10.0, 12.0, 11.0, 9.0, 13.0, 50.0, 60.0, 55.0],
                "label": ["human"] * 5 + ["bot"] * 3,
            },
            index=[f"u{i}" for i in range(8)],
        )
        humanlike_series, cfg = calibrate_circadian_humanlike(
            raw,
            label_col="label",
            honest_label="human",
            min_honest=5,
            mapping="gaussian",
        )
        # Gaussian peaks at honest median; bots far away should score near 0.
        self.assertAlmostEqual(cfg.target_z, 11.0, places=6)
        self.assertGreater(humanlike_series.loc["u2"], 0.95)
        self.assertLess(humanlike_series.loc["u5"], 0.05)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
