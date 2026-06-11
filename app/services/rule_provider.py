"""

RuleProvider abstraction: maps a language string to the Semgrep --config value.

Configuration is driven entirely by semgrep_rules.yaml:

    default: "auto"
    languages:
      python: "p/python"
      java:   "p/java"
      ...

To switch from community rules to custom rules for any language, update the
YAML file — zero code changes required.

Future example:
    python: "/app/rules/custom/python.yaml"

Singleton design
----------------
Problem (before fix):
    run_semgrep() called RuleProvider() on every Celery task.
    Each RuleProvider.__init__ called get_semgrep_rules_config() which,
    while lru_cached, still constructed a new RuleProvider object, allocated
    two new dicts (_languages, _default), and discarded them after one use.
    At lakh scale this is pure waste — the YAML never changes at runtime.

Solution:
    A single module-level instance `_RULE_PROVIDER` is built once when this
    module is first imported. All callers use `get_rule_provider()` to access
    it. Zero object construction overhead on every subsequent task.

    The YAML config itself is already lru_cached in config_loader.py, so
    reloading it is not a concern — this fix eliminates the object churn.
"""

from app.utils.config_loader import get_semgrep_rules_config
import logging

logger = logging.getLogger(__name__)


class RuleProvider:
    """
    Resolves the Semgrep --config argument for a given language.
    Intended to be used as a singleton via get_rule_provider().
    """

    def __init__(self) -> None:
        config = get_semgrep_rules_config()
        self._languages: dict[str, str] = config.get("languages", {})
        self._default: str = config.get("default", "auto")
        logger.debug(
            "RuleProvider initialised — %d language mappings, default=%s",
            len(self._languages),
            self._default,
        )

    def get_rules(self, language: str) -> str:
        """
        Return the Semgrep --config value for *language*.

        Falls back to the configured default (typically "auto") if the
        language is not listed in semgrep_rules.yaml.

        Args:
            language: Normalised language name (e.g. 'python', 'javascript').

        Returns:
            A string suitable for passing to `semgrep --config <value>`.
        """
        return self._languages.get(language.lower(), self._default)

    def supported_languages(self) -> list[str]:
        """Return the list of languages with explicit rule mappings."""
        return list(self._languages.keys())




_RULE_PROVIDER: RuleProvider = RuleProvider()


def get_rule_provider() -> RuleProvider:
    """
    Return the shared RuleProvider singleton.

    Always use this function instead of RuleProvider() directly.
    Callers in semgrep_runner.py and anywhere else get the same object
    every time with zero overhead.
    """
    return _RULE_PROVIDER