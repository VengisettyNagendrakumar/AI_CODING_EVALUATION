"""

Extracts complexity, structural, and quality metrics from a Tree-sitter AST.

All metrics are language-agnostic: they rely on node types that exist across
every language Tree-sitter supports (if/for/while/function_definition, etc.)
using a shared vocabulary of node-type sets defined below.

Extracted metrics
-----------------
- cyclomatic_complexity   Number of independent decision paths (McCabe metric).
- nesting_depth           Maximum depth of nested control-flow blocks.
- function_length         Maximum lines-of-code in any single function/method.
- parameter_count         Maximum number of parameters in any single function.

Cyclomatic complexity — design notes
-------------------------------------
McCabe's original definition (1976):
  CC = number_of_decision_points + 1

Decision points are nodes that introduce a NEW execution path:
  - if / elif / for / while / do / case / catch / except / ternary / match_arm

NOT a decision point:
  - else_clause  — this is the fallthrough of an already-counted `if`.
                   Counting it would double-penalise every if/else block.

Boolean short-circuit operators (&&, ||, and, or):
  Each operator token in a chained expression is its own decision point.
  `a && b && c` contains TWO operator tokens → +2.
  The old code used `break` after the first token → always +1 regardless
  of chain length. Fixed: count ALL operator tokens in the expression.

Iterative traversal — design notes
-------------------------------------
The old code used recursive _walk() functions. Python's default recursion
limit is ~1000 frames. A student submitting deeply nested generated code
(common in competitive programming or transpiled output) would hit this and
crash with RecursionError. All three traversal functions now use an explicit
stack (iterative DFS) instead of recursion — no stack overflow possible.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any

from app.services.ast_generator import ASTResult



_BRANCH_TYPES: frozenset[str] = frozenset({
    "if_statement",
    "elif_clause",
    # "else_clause" — excluded: see module docstring
    "for_statement",
    "for_in_statement",
    "while_statement",
    "do_statement",
    "case_statement",
    "switch_case",
    "catch_clause",
    "except_clause",
    "conditional_expression",   # ternary: a ? b : c
    "binary_expression",        # && and || short-circuits (counted per operator token)
    "boolean_operator",         # Python `and` / `or`   (counted per operator token)
    "ternary_expression",
    "when_expression",          # Kotlin/Scala when
    "match_arm",                # Rust match
})

# Operator token types that count as individual decision points inside
# binary_expression / boolean_operator nodes.
_BOOLEAN_OPERATOR_TOKENS: frozenset[str] = frozenset({
    "&&", "||", "and", "or",
})

# Control-flow container types used for nesting-depth tracking.
_NESTING_TYPES: frozenset[str] = frozenset({
    "if_statement",
    "for_statement",
    "for_in_statement",
    "while_statement",
    "do_statement",
    "try_statement",
    "with_statement",
    "match_statement",
    "switch_statement",
})

# Function/method definition node types.
_FUNCTION_TYPES: frozenset[str] = frozenset({
    "function_definition",
    "function_declaration",
    "method_definition",
    "method_declaration",
    "arrow_function",
    "lambda",
    "anonymous_function",
    "constructor_declaration",
    "function_expression",
    "local_function_statement",
    "fun_expression",           # Kotlin
})

# Parameter list node types.
_PARAM_LIST_TYPES: frozenset[str] = frozenset({
    "parameters",
    "parameter_list",
    "formal_parameters",
    "argument_list",
})

# Punctuation tokens to exclude when counting parameters.
_PARAM_PUNCTUATION: frozenset[str] = frozenset({",", "(", ")", " "})


# Result

@dataclass
class ASTMetrics:
    cyclomatic_complexity: int
    nesting_depth: int
    function_length: int
    parameter_count: int


# Public interface

def extract_metrics(ast_result: ASTResult) -> ASTMetrics:
    """
    Walk the Tree-sitter AST and extract all structural metrics.

    Args:
        ast_result: Output of ast_generator.generate_ast().

    Returns:
        ASTMetrics dataclass.
    """
    root = ast_result.tree.root_node
    lines = ast_result.code.splitlines()

    complexity = _cyclomatic_complexity(root)
    depth = _max_nesting_depth(root)
    fn_length, param_count = _function_metrics(root, lines)

    return ASTMetrics(
        cyclomatic_complexity=complexity,
        nesting_depth=depth,
        function_length=fn_length,
        parameter_count=param_count,
    )


# Internal traversal helpers — all iterative (no recursion)

def _cyclomatic_complexity(root: Any) -> int:
    """
    Count decision-point nodes to approximate McCabe cyclomatic complexity.
    Base complexity starts at 1 (for the entry path of the program).

    Uses iterative DFS to avoid Python's recursion limit on deeply nested code.

    Boolean operator handling:
        A binary_expression / boolean_operator node can contain multiple
        operator tokens: `a && b && c` has TWO `&&` tokens → +2.
        We count ALL operator tokens in the node, not just the first.
    """
    count = 1  # base path
    stack = deque([root])

    while stack:
        node = stack.pop()

        if node.type in _BRANCH_TYPES:
            if node.type in {"binary_expression", "boolean_operator"}:
                # Count every &&, ||, and, or token in this node.
                count += sum(
                    1 for child in node.children
                    if child.type in _BOOLEAN_OPERATOR_TOKENS
                )
            else:
                count += 1

        # Push children in reverse so left-to-right DFS order is preserved.
        stack.extend(reversed(node.children))

    return count


def _max_nesting_depth(root: Any) -> int:
    """
    Return the maximum nesting depth of control-flow blocks.
    Uses iterative DFS with explicit depth tracking to avoid recursion limit.
    """
    max_depth = 0
    # Stack entries: (node, current_depth)
    stack = deque([(root, 0)])

    while stack:
        node, current_depth = stack.pop()

        if node.type in _NESTING_TYPES:
            current_depth += 1
            if current_depth > max_depth:
                max_depth = current_depth

        # Push children in reverse to maintain left-to-right traversal.
        for child in reversed(node.children):
            stack.append((child, current_depth))

    return max_depth


def _function_metrics(root: Any, lines: list) -> tuple:
    """
    Walk the AST collecting function/method definitions and return:
      (max_function_length_in_lines, max_parameter_count)

    Uses iterative DFS to avoid recursion limit on deeply nested code.
    """
    max_length = 0
    max_params = 0
    stack = deque([root])

    while stack:
        node = stack.pop()

        if node.type in _FUNCTION_TYPES:
            # Function length = end line - start line + 1
            length = node.end_point[0] - node.start_point[0] + 1
            if length > max_length:
                max_length = length

            # Parameter count: find the parameter list child.
            for child in node.children:
                if child.type in _PARAM_LIST_TYPES:
                    param_count = sum(
                        1
                        for p in child.children
                        if p.type not in _PARAM_PUNCTUATION
                        and not p.is_extra   
                    )
                    if param_count > max_params:
                        max_params = param_count
                    break

        stack.extend(reversed(node.children))

    return max_length, max_params