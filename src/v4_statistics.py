"""Shared uncertainty calculations for the public-data v4 audit.

The v4 extension uses one implementation for descriptive proportions and
incidence rates so that quality audits, status summaries, and manuscript
tables cannot silently use different interval conventions.
"""

from __future__ import annotations

import math

from scipy.stats import chi2, norm


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Return a Wilson score interval for a binomial proportion."""
    if total <= 0:
        return math.nan, math.nan
    if successes < 0 or successes > total:
        raise ValueError("successes must be between zero and total")
    alpha = 1.0 - confidence
    z = float(norm.ppf(1.0 - alpha / 2.0))
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def poisson_rate_interval(
    events: int,
    exposure: float,
    scale: float = 1.0,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return an exact central Poisson interval for a scaled incidence rate."""
    if events < 0:
        raise ValueError("events must be non-negative")
    if exposure <= 0 or scale <= 0:
        return math.nan, math.nan
    alpha = 1.0 - confidence
    lower_count = 0.0 if events == 0 else 0.5 * float(chi2.ppf(alpha / 2.0, 2 * events))
    upper_count = 0.5 * float(chi2.ppf(1.0 - alpha / 2.0, 2 * (events + 1)))
    return lower_count / exposure * scale, upper_count / exposure * scale


def percent_with_interval(successes: int, total: int) -> tuple[float, float, float]:
    """Return a percentage and its 95% Wilson interval in percentage units."""
    if total <= 0:
        return math.nan, math.nan, math.nan
    low, high = wilson_interval(successes, total)
    return successes / total * 100.0, low * 100.0, high * 100.0
