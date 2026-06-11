"""

Language detection and validation using Tree-sitter 0.22+ API.

Strategy:
  - The client declares a language in the request body.
  - We load the corresponding tree-sitter language module and parse the code.
  - If the parse produces a clean tree (root is not mostly ERROR nodes),
    the language is validated.
  - If the parse fails or the grammar package is missing, a clear error
    is raised so the API can return a descriptive response.

We deliberately do NOT auto-detect or guess the language.

Parser caching — design notes
------------------------------
Problem (before fix):
  _load_parser() called importlib.import_module() + Language() + Parser()
  on EVERY request. For a system handling lakhs of students this means:
    - The same grammar binary is re-imported thousands of times per second.
    - GC pressure from throwaway Language and Parser objects.
    - CPU time wasted on module lookup and object construction.

Solution:
  _LANGUAGE_CACHE stores the built Language object per language name.
  Language objects are safe to share across threads — they are read-only
  wrappers around the compiled grammar binary.

  Parser objects are NOT cached and NOT shared. Tree-sitter's Parser
  is stateful during a parse call — sharing one Parser across concurrent
  threads will corrupt parse state. Instead:
    - Language  → cached at module level (one per language, forever)
    - Parser    → created fresh per request  (cheap: just wraps the Language)

  This gives us:
    - Zero reimports after the first request per language.
    - Full thread safety under concurrent uvicorn workers.
    - Parser construction cost is O(1) — just a Python object wrapping
      an already-loaded C struct, not a grammar reload.
"""

import importlib
import logging
import threading
from dataclasses import dataclass
from typing import Any

from tree_sitter import Language, Parser

logger = logging.getLogger(__name__)


# Language registry
# Maps normalised language name → (module_name, language_fn)
# language_fn is the function inside the module that returns the language capsule.
# Most packages export language(), but typescript exports language_typescript().

_LANGUAGE_REGISTRY: dict[str, tuple[str, str]] = {
    "python":       ("tree_sitter_python",      "language"),
    "javascript":   ("tree_sitter_javascript",  "language"),
    "typescript":   ("tree_sitter_typescript",  "language_typescript"),
    "java":         ("tree_sitter_java",        "language"),
    "go":           ("tree_sitter_go",          "language"),
    "rust":         ("tree_sitter_rust",        "language"),
    "cpp":          ("tree_sitter_cpp",         "language"),
    "c":            ("tree_sitter_c",           "language"),
    "sql":          ("tree_sitter_sql",         "language"),
}


# Language alias normalisation

_ALIASES: dict[str, str] = {
    "js":       "javascript",
    "ts":       "typescript",
    "py":       "python",
    "rs":       "rust",
    "c++":      "cpp",
    "golang":   "go",
    "c#":       "c_sharp",
    "csharp":   "c_sharp",
    "cs":       "c_sharp",
    "rb":       "ruby",
    "kt":       "kotlin",
}


def _normalise(language: str) -> str:
    """Return the canonical registry key for a language string."""
    return _ALIASES.get(language.lower().strip(), language.lower().strip())


# Custom exceptions

class LanguageNotSupportedError(ValueError):
    """Raised when no grammar package is registered for the declared language."""


class LanguageMismatchError(ValueError):
    """Raised when the code cannot be parsed under the declared language grammar."""




_LANGUAGE_CACHE: dict[str, Language] = {}
_CACHE_LOCK = threading.Lock()


def _get_language(normalised: str) -> Language:
    """
    Return the cached tree-sitter Language for *normalised*.
    Loads and caches it on first call; returns the cached object on all
    subsequent calls with zero import/construction overhead.

    Thread-safe: uses a lock only during the first load of each language.
    """
    # Fast path — language already cached (no lock needed after first load).
    if normalised in _LANGUAGE_CACHE:
        return _LANGUAGE_CACHE[normalised]

    # Slow path — first time this language is requested.
    with _CACHE_LOCK:
        # Double-checked locking: another thread may have loaded it while
        # we were waiting for the lock.
        if normalised in _LANGUAGE_CACHE:
            return _LANGUAGE_CACHE[normalised]

        entry = _LANGUAGE_REGISTRY.get(normalised)
        if not entry:
            raise LanguageNotSupportedError(
                f"Language '{normalised}' is not in the supported language registry. "
                f"Supported: {sorted(_LANGUAGE_REGISTRY.keys())}"
            )

        module_name, fn_name = entry
        try:
            module   = importlib.import_module(module_name)
            lang_fn  = getattr(module, fn_name)
            language = Language(lang_fn())
            _LANGUAGE_CACHE[normalised] = language
            logger.info(
                "Grammar loaded and cached — language=%s module=%s",
                normalised, module_name,
            )
            return language

        except ImportError:
            raise LanguageNotSupportedError(
                f"Grammar package '{module_name}' is not installed. "
                f"Run: pip install {module_name.replace('_', '-')}"
            )
        except Exception as exc:
            logger.exception("Failed to load grammar for '%s': %s", normalised, exc)
            raise LanguageNotSupportedError(
                f"Failed to load grammar for '{normalised}': {exc}"
            )


def _build_parser(language: Language) -> Parser:
    """
    Build a fresh Parser for the given Language.

    Called once per request — cheap because it only wraps the already-cached
    Language object. NOT cached because Parser is stateful during parse()
    and must not be shared across concurrent threads/requests.
    """
    return Parser(language)


# Public interface

@dataclass
class DetectionResult:
    language: str    # normalised registry key
    declared: str    # original string from the client
    parser: Any      # ready-to-use tree_sitter.Parser (fresh per request)


def detect_and_validate(declared_language: str, code: str) -> "DetectionResult":
    """
    Validate that *code* is parseable as *declared_language*.

    On first call for a language: imports grammar, builds Language, caches it.
    On subsequent calls:          returns cached Language, builds fresh Parser.

    Returns a DetectionResult on success.
    Raises LanguageNotSupportedError if the grammar package is unavailable.
    Raises LanguageMismatchError if the parse produces an error tree.
    """
    normalised = _normalise(declared_language)

    # Language loaded from cache (or imported once and cached).
    language = _get_language(normalised)

    # Fresh Parser per request — thread-safe, negligible cost.
    parser = _build_parser(language)

    # Parse and validate the tree.
    tree = parser.parse(bytes(code, "utf-8"))
    root = tree.root_node

    if root.type == "ERROR" or (root.has_error and _root_is_mostly_error(root)):
        raise LanguageMismatchError(
            f"Code does not appear to be valid {declared_language}. "
            f"Tree-sitter produced an error tree. "
            f"Check that 'language' matches the submitted code."
        )

    return DetectionResult(language=normalised, declared=declared_language, parser=parser)


def _root_is_mostly_error(root: Any) -> bool:
    """
    Return True if most of the root's direct children contain errors.
    Avoids false positives on code with minor syntax issues.
    """
    if root.child_count == 0:
        return False
    error_children = sum(1 for c in root.children if c.type == "ERROR" or c.has_error)
    return (error_children / root.child_count) > 0.5