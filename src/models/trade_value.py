"""Tier-based mapping from OVR (0-100) to dollar trade value.

Stage 2 / Option A: deterministic. Breakpoints are anchored to the
2025-26 NBA contract structure:
  * supermax (35% cap):       ~$55M
  * 25-30% max:              ~$38-46M
  * Mid-Level Exception:      ~$13M
  * Veteran minimum:          ~$2.4M

Linear interpolation between breakpoints gives a smooth curve while the
named tiers ("Superstar", "All-Star", etc.) classify the OVR for readability.
"""
from __future__ import annotations

import numpy as np

# Anchor points: (OVR, fair-market salary in $). Interpolated linearly between.
OVR_BREAKPOINTS = [0, 40, 50, 60, 65, 70, 75, 80, 85, 90, 95, 100]
SALARY_BREAKPOINTS = [
    500_000,    # below replacement level
    2_400_000,  # vet minimum
    3_000_000,  # bench
    5_000_000,
    9_000_000,
    13_000_000,  # MLE-tier rotation player
    18_000_000,
    24_000_000,
    32_000_000,  # All-Star
    42_000_000,
    50_000_000,  # superstar
    55_000_000,  # supermax ceiling
]

TIERS: list[tuple[float, str]] = [
    (95, "Superstar"),
    (85, "All-Star"),
    (75, "Quality Starter"),
    (65, "Rotation"),
    (50, "Bench"),
    (0,  "Marginal"),
]


def ovr_to_dollars(ovr: float) -> float:
    """Linear-interpolated fair-market salary for a given OVR. Returns $ as float."""
    if ovr is None or (isinstance(ovr, float) and np.isnan(ovr)):
        return float("nan")
    clipped = float(np.clip(ovr, 0, 100))
    return float(np.interp(clipped, OVR_BREAKPOINTS, SALARY_BREAKPOINTS))


def ovr_to_tier(ovr: float) -> str:
    """Return the descriptive tier label for a given OVR."""
    if ovr is None or (isinstance(ovr, float) and np.isnan(ovr)):
        return "?"
    for threshold, name in TIERS:
        if ovr >= threshold:
            return name
    return "Marginal"


def format_dollars(amount: float) -> str:
    """Format a dollar amount as '$42.3M' or '$2.4M' or '$500K'."""
    if amount is None or (isinstance(amount, float) and np.isnan(amount)):
        return "$?"
    if amount >= 1_000_000:
        return f"${amount / 1_000_000:.1f}M"
    return f"${amount / 1_000:.0f}K"
