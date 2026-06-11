"""

Loads and caches YAML configuration files from the app/config/ directory.
All configs are loaded once per process and reused across all requests.

Caching strategy — lru_cache (not Redis)
-----------------------------------------
These YAML configs are:
  - Read-only  : they never change at runtime.
  - Tiny       : three small dicts, a few hundred bytes total.
  - Process-local : every worker needs its own RAM copy; no sharing needed.

lru_cache stores the parsed Python dict in process memory.
Access cost after first load: ~nanoseconds (a single dict lookup).

Redis cache would be wrong here:
  - Each read would cost 1-5ms (network round trip + JSON deserialize).
  - Redis becoming unavailable would break config reads entirely.
  - The data is not shared state — it never changes, every process
    needs the same copy, there is nothing to coordinate.

Startup validation
------------------
The three @lru_cache functions are lazy — they only load on first call.
Without explicit startup validation, a YAML syntax error surfaces as a
500 mid-request during a student's submission instead of at deploy time.

validate_configs_on_startup() force-loads all three configs immediately
when the FastAPI process starts. A bad YAML now fails fast and loud at
startup, not silently mid-exam.
"""

import logging
from pathlib import Path
from functools import lru_cache
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Resolve the config directory relative to this file's location.
_CONFIG_DIR = Path(__file__).parent.parent / "config"


# Internal loader

def _load_yaml(filename: str) -> Any:
    """
    Read and parse a YAML file from the config directory.

    Raises:
        FileNotFoundError: if the file does not exist.
        yaml.YAMLError:    if the file contains invalid YAML syntax.
    """
    path = _CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. "
            f"Ensure '{filename}' exists in {_CONFIG_DIR}."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(
            f"Config file '{filename}' is empty or contains only comments. "
            f"Ensure it has valid YAML content."
        )

    return data


# Cached config accessors
# lru_cache(maxsize=None): cache never evicts — configs are loaded exactly
# once per process lifetime and reused for every subsequent call.

@lru_cache(maxsize=None)
def get_normalization_rules() -> dict:
    """Return the normalization threshold tables keyed by metric name."""
    rules = _load_yaml("normalization_rules.yaml")
    logger.info("normalization_rules.yaml loaded — %d metrics defined", len(rules))
    return rules


@lru_cache(maxsize=None)
def get_scoring_weights() -> dict:
    """Return the weighted scoring coefficients."""
    weights = _load_yaml("scoring_weights.yaml")
    logger.info("scoring_weights.yaml loaded — %d weights defined", len(weights))
    return weights


@lru_cache(maxsize=None)
def get_semgrep_rules_config() -> dict:
    """Return the RuleProvider language-to-rules-path mapping."""
    config = _load_yaml("semgrep_rules.yaml")
    logger.info(
        "semgrep_rules.yaml loaded — %d language mappings, default=%s",
        len(config.get("languages", {})),
        config.get("default", "auto"),
    )
    return config


@lru_cache(maxsize=None)
def get_llm_scoring_weights() -> dict:
    """Return the LLM evaluation weighted scoring coefficients."""
    config = _load_yaml("llm_scoring_weights.yaml")
    weights = config.get("weights", {})
    logger.info("llm_scoring_weights.yaml loaded — %d weights defined", len(weights))
    return weights


@lru_cache(maxsize=None)
def get_aggregation_weights() -> dict:
    """Return the pipeline evaluation weighted scoring and capping config."""
    config = _load_yaml("aggregation_weights.yaml")
    logger.info("aggregation_weights.yaml loaded")
    return config


# Startup validation


def validate_configs_on_startup() -> None:
    """
    Force-load and validate all YAML configs at process startup.

    Call this from FastAPI's @app.on_event("startup") handler so that:
      - Missing files       → FileNotFoundError  at startup, not mid-request.
      - Empty files         → ValueError          at startup, not mid-request.
      - Invalid YAML syntax → yaml.YAMLError      at startup, not mid-request.
      - Wrong weight sum    → ValueError          at startup, not mid-request.

    All lru_cache functions are populated here, so the very first
    real request hits the cache immediately with zero disk I/O.

    Raises:
        FileNotFoundError: if any config file is missing.
        ValueError:        if a config file is empty or weights don't sum to 1.0.
        yaml.YAMLError:    if any config file has invalid YAML syntax.
    """
    logger.info("Validating YAML configs at startup...")

    # Force-load all configs — exceptions propagate immediately.
    get_normalization_rules()
    get_scoring_weights()
    get_semgrep_rules_config()
    get_llm_scoring_weights()
    get_aggregation_weights()

    # Validate all weight configs sum to 1.0 at startup, not mid-request.
    _validate_scoring_weights()
    _validate_llm_scoring_weights()
    _validate_aggregation_weights()

    logger.info("All YAML configs validated successfully.")


def _validate_scoring_weights() -> None:
    """
    Validate that scoring_weights.yaml has all required keys and sums to 1.0.
    Called only from validate_configs_on_startup() — runs once at boot.

    Raises:
        KeyError:   if a required weight key is missing.
        ValueError: if weights do not sum to 1.0.
    """
    required_keys = {
        "maintainability", "security", "complexity",
        "reliability", "best_practices",
    }
    weights = get_scoring_weights()

    missing = required_keys - weights.keys()
    if missing:
        raise KeyError(
            f"scoring_weights.yaml is missing required keys: {sorted(missing)}. "
            f"Add them and ensure all weights sum to 1.0."
        )

    total = sum(weights[k] for k in required_keys)
    if abs(total - 1.0) > 0.001:
        raise ValueError(
            f"scoring_weights.yaml weights must sum to 1.0, but got {total:.4f}. "
            f"Current values: { {k: weights[k] for k in required_keys} }. "
            f"Adjust the weights so they sum to exactly 1.0."
        )


def _validate_llm_scoring_weights() -> None:
    """
    Validate that llm_scoring_weights.yaml has all required keys and sums to 1.0.
    Called only from validate_configs_on_startup() — runs once at boot.

    Raises:
        KeyError:   if a required weight key is missing.
        ValueError: if weights do not sum to 1.0.
    """
    required_keys = {
        "correctness", "logic", "optimization",
        "edge_case_handling", "readability",
    }
    weights = get_llm_scoring_weights()

    missing = required_keys - weights.keys()
    if missing:
        raise KeyError(
            f"llm_scoring_weights.yaml is missing required keys: {sorted(missing)}. "
            f"Add them and ensure all weights sum to 1.0."
        )

    total = sum(weights[k] for k in required_keys)
    if abs(total - 1.0) > 0.001:
        raise ValueError(
            f"llm_scoring_weights.yaml weights must sum to 1.0, but got {total:.4f}. "
            f"Current values: { {k: weights[k] for k in required_keys} }. "
            f"Adjust the weights so they sum to exactly 1.0."
        )


def _validate_aggregation_weights() -> None:
    """
    Validate that aggregation_weights.yaml has all required keys and sums to 1.0.
    """
    config = get_aggregation_weights()
    weights = config.get("weights", {})
    caps = config.get("caps", {})

    required_weight_keys = {
        "static_analysis", "llm_evaluation", "runtime_execution",
    }
    missing_weights = required_weight_keys - weights.keys()
    if missing_weights:
        raise KeyError(
            f"aggregation_weights.yaml is missing required weights keys: {sorted(missing_weights)}."
        )

    required_caps_keys = {
        "syntax_failure_cap", "compilation_failure_cap", "zero_tests_passed_cap",
    }
    missing_caps = required_caps_keys - caps.keys()
    if missing_caps:
        raise KeyError(
            f"aggregation_weights.yaml is missing required caps keys: {sorted(missing_caps)}."
        )

    total = sum(weights[k] for k in required_weight_keys)
    if abs(total - 1.0) > 0.001:
        raise ValueError(
            f"aggregation_weights.yaml weights must sum to 1.0, but got {total:.4f}."
        )