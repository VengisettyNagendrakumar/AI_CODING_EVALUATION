"""

Runs Semgrep against a code submission and returns categorised finding counts.

Pipeline:
  1. RuleProvider resolves the correct --config value for the language.
  2. The code is written to a temporary file (Semgrep requires a file path).
  3. Semgrep is invoked as a subprocess with --json output.
  4. JSON output is parsed and findings are categorised by severity:
       ERROR   → security_issues
       WARNING → reliability_issues
       INFO    → best_practice_violations
  5. The temp file is deleted in a finally block.

Semgrep failure handling
------------------------
Previously: any Semgrep failure (not installed, crash, timeout) silently
returned zero findings. Zero findings → security_score=100, reliability_score=100.
A student with terrible code could get boosted scores because Semgrep failed.

Fix: SemgrepResult now carries a `semgrep_available` flag.
  - semgrep_available=True  → real findings, use scores normally
  - semgrep_available=False → Semgrep failed, mark static analysis as degraded

The tasks.py and scorer.py layers use this flag to:
  - Surface the failure in the result (analysis_status="degraded")
  - Skip Semgrep-derived dimension scores rather than treating zero as perfect
"""

import json
import subprocess
import tempfile
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path

from app.services.rule_provider import get_rule_provider

logger = logging.getLogger(__name__)

_OK_EXIT_CODES = {0, 1}
_SEMGREP_TIMEOUT_NORMAL = 30
_SEMGREP_TIMEOUT_LARGE  = 60
_SEMGREP_MAX_MEMORY_MB  = 1024

_EXT_MAP: dict[str, str] = {
    "python":     ".py",
    "javascript": ".js",
    "typescript": ".ts",
    "java":       ".java",
    "go":         ".go",
    "rust":       ".rs",
    "cpp":        ".cpp",
    "c":          ".c",
    "c_sharp":    ".cs",
    "ruby":       ".rb",
    "php":        ".php",
    "kotlin":     ".kt",
    "swift":      ".swift",
    "scala":      ".scala",
}


@dataclass
class SemgrepResult:
    security_issues: int
    reliability_issues: int
    best_practice_violations: int
    # NEW: False when Semgrep failed — callers must not treat zero as "no issues"
    semgrep_available: bool = True
    # NEW: Human-readable reason when semgrep_available=False
    failure_reason: str = ""


def run_semgrep(language: str, code: str, large_submission: bool = False) -> SemgrepResult:
    """
    Execute Semgrep on *code* and return categorised finding counts.

    Returns SemgrepResult with semgrep_available=False if Semgrep is not
    installed or fails — callers must check this flag and surface the
    degraded state rather than silently treating zeros as clean results.
    """
    rules_config = get_rule_provider().get_rules(language)
    ext     = _extension_for(language)
    timeout = _SEMGREP_TIMEOUT_LARGE if large_submission else _SEMGREP_TIMEOUT_NORMAL

    if large_submission:
        logger.info(
            "Running Semgrep with extended limits — language=%s timeout=%ds max_memory=%dMB",
            language, timeout, _SEMGREP_MAX_MEMORY_MB,
        )

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=ext, delete=False, encoding="utf-8",
        ) as tmp:
            tmp.write(code)
            tmp_path = Path(tmp.name)

        result = _invoke_semgrep(rules_config, tmp_path, timeout)
        return _parse_output(result)

    except FileNotFoundError:
        reason = (
            "Semgrep is not installed. "
            "Install with: pip install semgrep"
        )
        logger.warning(
            "Semgrep not found — static security/reliability scores unavailable. %s",
            reason,
        )
        return SemgrepResult(
            security_issues=0,
            reliability_issues=0,
            best_practice_violations=0,
            semgrep_available=False,
            failure_reason=reason,
        )
    except Exception as exc:
        reason = f"Semgrep analysis failed: {exc}"
        logger.warning(
            "Semgrep failed — static security/reliability scores unavailable. %s",
            reason,
        )
        return SemgrepResult(
            security_issues=0,
            reliability_issues=0,
            best_practice_violations=0,
            semgrep_available=False,
            failure_reason=reason,
        )
    finally:
        if tmp_path and tmp_path.exists():
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _invoke_semgrep(rules_config: str, target: Path, timeout: int) -> dict:
    cmd = [
        "semgrep",
        "--config", rules_config,
        "--json",
        "--quiet",
        "--no-git-ignore",
        f"--timeout={timeout}",
        f"--max-memory={_SEMGREP_MAX_MEMORY_MB}",
        str(target),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout + 10,
    )

    if proc.returncode not in _OK_EXIT_CODES:
        logger.warning(
            "Semgrep exited with code %d. stderr: %s",
            proc.returncode, proc.stderr[:500],
        )
        return {}

    if not proc.stdout.strip():
        return {}

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse Semgrep JSON output: %s", exc)
        return {}


def _parse_output(semgrep_json: dict) -> SemgrepResult:
    security = reliability = best_practice = 0
    for finding in semgrep_json.get("results", []):
        severity: str = finding.get("extra", {}).get("severity", "").upper()
        if severity == "ERROR":
            security += 1
        elif severity == "WARNING":
            reliability += 1
        elif severity in {"INFO", "INVENTORY", "EXPERIMENT"}:
            best_practice += 1
        else:
            best_practice += 1

    return SemgrepResult(
        security_issues=security,
        reliability_issues=reliability,
        best_practice_violations=best_practice,
        semgrep_available=True,
    )


def _extension_for(language: str) -> str:
    return _EXT_MAP.get(language, ".txt")