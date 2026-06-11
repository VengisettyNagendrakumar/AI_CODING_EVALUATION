"""

Generates a Tree-sitter syntax tree from source code using the 0.22+ API.

The Parser is received from DetectionResult (already built during language
validation) to avoid loading the grammar twice per request.
"""

from dataclasses import dataclass
from typing import Any

from app.services.lang_detector import DetectionResult


@dataclass
class ASTResult:
    tree: Any       # tree_sitter.Tree
    language: str   # normalised grammar name used for parsing
    code: str       # original source (needed for line-based metrics)


def generate_ast(detection: DetectionResult, code: str) -> ASTResult:
    """
    Parse *code* using the already-loaded Parser from *detection*.

    Args:
        detection: DetectionResult from lang_detector.detect_and_validate().
        code:      Raw source code string.

    Returns:
        ASTResult with the parsed tree, language name, and original code.
    """
    tree = detection.parser.parse(bytes(code, "utf-8"))
    return ASTResult(tree=tree, language=detection.language, code=code)
