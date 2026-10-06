"""Synthetic checks of the R1 statistics (review PCR-006): ties, zeros, all-zero, reverse effects, Holm, integer scale.

    python r1/test_aggregate_stats.py      (exit 0 = all checks pass)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r1.aggregate_r1 import holm, int_stats, label, rank_biserial  # noqa: E402

N_TEST = 10000
fails = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


# rank-biserial with tied |d| and a zero: |d| = 1,1,1,2 -> ranks 2,2,2,4; W+ = 8, W- = 2 -> (8-2)/10
check("rank_biserial ties+zero", abs(rank_biserial([1, 1, -1, 2, 0]) - 0.6) < 1e-12)
check("rank_biserial all positive", rank_biserial([3, 1, 2]) == 1.0)
check("rank_biserial all zero", rank_biserial([0, 0]) == 0.0)

# all-zero differences
z = int_stats([0] * 20, 1, N_TEST)
check("all_zero p=1", z["p_two_sided"] == 1.0 and z["wilcoxon_method"] == "all_zero")
check("all_zero d_z undefined", z["d_z"] == "undefined")
check("all_zero label", label(1.0, z) == "no_difference")

# clear positive effect without ties -> exact method, supported
pos = int_stats(list(range(5, 25)), 2, N_TEST)
check("positive exact method", pos["wilcoxon_method"] == "exact")
check("positive supported", label(pos["p_two_sided"], pos) == "supported")
check("positive pp scale", abs(pos["mean_diff_pp"] - 100 * 14.5 / N_TEST) < 1e-12)

# clear negative effect -> reverse, never 'supported' (review PCR-001)
neg = int_stats([-x for x in range(5, 25)], 3, N_TEST)
check("negative reverse", label(neg["p_two_sided"], neg) == "reverse")

# ties and zeros -> approx method
tz = int_stats([0, 2, 2, 3, 3, 3, -1, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 12, 0], 4, N_TEST)
check("ties/zeros approx method", tz["wilcoxon_method"] == "approx")

# constant non-zero difference -> d_z undefined but test defined
const = int_stats([3] * 20, 5, N_TEST)
check("constant difference d_z undefined", const["d_z"] == "undefined" and const["wilcoxon_method"] == "approx")

# Holm: monotone, capped, order-preserving
adj = holm([0.01, 0.04, 0.03, 0.5])
check("holm values", [round(x, 10) for x in adj] == [0.04, 0.09, 0.09, 0.5])
check("holm cap", max(holm([0.6, 0.7])) <= 1.0)

# integer scale: the float pitfall of the review example disappears on counts
a, b, c = round(0.9 * N_TEST), round(0.8 * N_TEST), round(0.7 * N_TEST)
check("integer differences equal", (a - b) == (b - c) == 1000)

# deterministic bootstrap
check("bootstrap deterministic", int_stats(list(range(1, 21)), 7, N_TEST) == int_stats(list(range(1, 21)), 7, N_TEST))

print("FAILED:", fails if fails else "none")
sys.exit(1 if fails else 0)
