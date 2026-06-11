"""

Converts raw metric values to normalised 0–100 scores using the threshold
tables defined in normalization_rules.yaml.

Normalisation algorithm (threshold-based lookup):
  Given a list of (max, score) buckets in ascending order of max:
  - Walk through each bucket.
  - Return the score of the first bucket whose max >= raw_value.
  - If the raw value exceeds all defined maxes, return the last bucket's score.

This approach is transparent, easily tuneable, and requires no statistical
calibration on historical data.
"""

import logging
from app.utils.config_loader import get_normalization_rules

logger = logging.getLogger(__name__)


def normalize(metric_name: str, raw_value: int | float) -> float:
    """
    Normalise a single raw metric value to a 0–100 score.

    Args:
        metric_name: Key in normalization_rules.yaml (e.g. 'cyclomatic_complexity').
        raw_value:   The raw metric value to normalise.

    Returns:
        A float score in [0, 100].
        Returns 50.0 as a neutral fallback if the metric is not found in config.
    """
    rules = get_normalization_rules()
    buckets = rules.get(metric_name)

    if not buckets:
        logger.warning(
            "No normalisation rule found for metric '%s'. Returning neutral score 50.",
            metric_name,
        )
        return 50.0

    for bucket in buckets:
        if raw_value <= bucket["max"]:
            return float(bucket["score"])

    # Exceeded all defined buckets — return the last (lowest) score.
    return float(buckets[-1]["score"])
