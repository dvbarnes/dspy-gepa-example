import dspy
from typing import Optional, Any
from datetime import datetime
from rapidfuzz import fuzz
from gepa import ScoreWithFeedback


# ─────────────────────────────────────────────
# Field-level scoring helpers
# Each returns (score: float, issue: str)
# ─────────────────────────────────────────────

def _score_date(gold_val: Any, pred_val: Any) -> tuple[float, str]:
    if gold_val is None and pred_val is None:
        return 1.0, ""
    if pred_val is None:
        return 0.0, f"Date missing entirely (expected '{gold_val}')."

    gold_str, pred_str = str(gold_val).strip(), str(pred_val).strip()

    if gold_str == pred_str:
        return 1.0, ""

    # Normalize common OCR/format variants and compare
    formats = ("%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%B %d, %Y", "%b %d, %Y", "%m-%d-%Y")
    gold_dt = pred_dt = None
    for fmt in formats:
        try:
            gold_dt = datetime.strptime(gold_str, fmt)
        except ValueError:
            pass
        try:
            pred_dt = datetime.strptime(pred_str, fmt)
        except ValueError:
            pass

    if gold_dt and pred_dt:
        if gold_dt == pred_dt:
            # Correct date, wrong format
            return 0.8, f"Date value correct but wrong format: got '{pred_str}', expected '{gold_str}' (use YYYY-MM-DD)."
        else:
            return 0.0, f"Date value wrong: got '{pred_str}', expected '{gold_str}'."

    return 0.0, f"Date could not be parsed: got '{pred_str}', expected '{gold_str}'."


def _score_text(
    gold_val: Any,
    pred_val: Any,
    field: str,
    partial: bool = False,
) -> tuple[float, str]:
    if gold_val is None and pred_val is None:
        return 1.0, ""
    if pred_val is None:
        return 0.0, f"{field.capitalize()} missing entirely (expected '{gold_val}')."

    g = str(gold_val).lower().strip()
    p = str(pred_val).lower().strip()

    if g == p:
        return 1.0, ""

    score = (fuzz.partial_ratio(g, p) if partial else fuzz.token_sort_ratio(g, p)) / 100.0

    if score >= 0.95:
        return score, ""
    if score >= 0.85:
        return score, f"{field.capitalize()} minor mismatch: got '{pred_val}', expected '{gold_val}'."
    if score >= 0.5:
        return score, f"{field.capitalize()} partial match: got '{pred_val}', expected '{gold_val}'."
    return score, f"{field.capitalize()} substantially wrong: got '{pred_val}', expected '{gold_val}'."


def _score_numeric(
    gold_val: Any,
    pred_val: Any,
    tolerance: float = 0.01,
) -> tuple[float, str]:
    if gold_val is None and pred_val is None:
        return 1.0, ""
    if pred_val is None:
        return 0.0, f"Amount missing entirely (expected {gold_val})."

    try:
        g = float(gold_val)
        p = float(pred_val)
    except (ValueError, TypeError):
        match = str(gold_val).strip() == str(pred_val).strip()
        return (1.0, "") if match else (
            0.0, f"Amount unparseable: got '{pred_val}', expected '{gold_val}'."
        )

    if g == 0 and p == 0:
        return 1.0, ""
    if g == 0:
        return 0.0, f"Amount wrong: got {p}, expected 0."

    rel_error = abs(g - p) / abs(g)
    if rel_error <= tolerance:
        return 1.0, ""
    if rel_error <= 0.05:
        return 0.5, f"Amount close but off: got {p}, expected {g} ({rel_error*100:.1f}% error)."
    return 0.0, f"Amount wrong: got {p}, expected {g} ({rel_error*100:.1f}% error)."


def _score_row(gold: dict, pred: dict) -> tuple[float, list[str]]:
    """Score a single row across all fields, return (score, issues)."""
    results = [
        _score_date(gold.get("date"), pred.get("date")),
        _score_text(gold.get("description"), pred.get("description"), "description"),
        _score_text(gold.get("location"), pred.get("location"), "location", partial=True),
        _score_numeric(gold.get("amount"), pred.get("amount")),
    ]
    scores, issues = zip(*results)
    return sum(scores) / len(scores), [i for i in issues if i]


def _summarize_issues(all_row_issues: list[tuple[int, list[str]]]) -> str:
    """
    Collapse per-row issues into pattern-level feedback for the reflection LM.
    Surfaces recurring failure types rather than listing every row individually.
    """
    date_issues, amount_issues, text_issues, missing_rows = [], [], [], []

    for row_idx, issues in all_row_issues:
        for issue in issues:
            low = issue.lower()
            if "date" in low:
                date_issues.append(f"row {row_idx+1}: {issue}")
            elif "amount" in low:
                amount_issues.append(f"row {row_idx+1}: {issue}")
            elif "missing entirely" in low and "row" not in low:
                missing_rows.append(f"row {row_idx+1}: {issue}")
            else:
                text_issues.append(f"row {row_idx+1}: {issue}")

    parts = []
    if date_issues:
        parts.append(
            f"DATE ERRORS ({len(date_issues)}): Always return dates as YYYY-MM-DD. "
            + "; ".join(date_issues[:3])
            + ("..." if len(date_issues) > 3 else "")
        )
    if amount_issues:
        parts.append(
            f"AMOUNT ERRORS ({len(amount_issues)}): Return amounts as plain numbers (no currency symbols). "
            + "; ".join(amount_issues[:3])
            + ("..." if len(amount_issues) > 3 else "")
        )
    if text_issues:
        parts.append(
            f"TEXT ERRORS ({len(text_issues)}): "
            + "; ".join(text_issues[:3])
            + ("..." if len(text_issues) > 3 else "")
        )
    if missing_rows:
        parts.append(
            f"MISSING FIELDS ({len(missing_rows)}): "
            + "; ".join(missing_rows[:3])
        )

    return "\n".join(parts) if parts else ""


# ─────────────────────────────────────────────
# GEPA Metric
# ─────────────────────────────────────────────

def table_extraction_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: Optional[Any] = None,
    pred_name: Optional[str] = None,
    pred_trace: Optional[Any] = None,
) -> "float | ScoreWithFeedback":
    """
    GEPA-compatible metric for image table extraction.

    - Returns ScoreWithFeedback (score + feedback string) when called by the
      GEPA optimizer (pred_name is set) so the reflection LM has actionable signal.
    - Returns plain float when called by dspy.Evaluate() or during normal inference.

    Scoring weights:
        20% row count accuracy
        80% field-level F1 across aligned rows
    """
    gold_rows: list[dict] = getattr(gold, "rows", []) or []
    pred_rows: list[dict] = getattr(pred, "rows", []) or []

    # ── Empty document case ──
    if not gold_rows and not pred_rows:
        if pred_name:
            return ScoreWithFeedback(
                score=1.0,
                feedback="Correct: document has no entries and none were returned."
            )
        return 1.0

    if not gold_rows and pred_rows:
        score = 0.0
        if pred_name:
            return ScoreWithFeedback(
                score=score,
                feedback=(
                    f"HALLUCINATION: Document has no entries but {len(pred_rows)} row(s) were returned. "
                    f"When a table is empty or absent, return an empty list."
                )
            )
        return score

    if not pred_rows:
        score = 0.0
        if pred_name:
            return ScoreWithFeedback(
                score=score,
                feedback=(
                    f"No rows extracted. Expected {len(gold_rows)} row(s). "
                    f"Ensure all visible table rows are captured, including single-row tables."
                )
            )
        return score

    # ── Row count score ──
    n_gold, n_pred = len(gold_rows), len(pred_rows)
    row_count_score = 1.0 - abs(n_gold - n_pred) / max(n_gold, n_pred)

    count_feedback = ""
    if n_pred < n_gold:
        count_feedback = f"ROW COUNT: {n_gold - n_pred} row(s) missed. Expected {n_gold}, got {n_pred}."
    elif n_pred > n_gold:
        count_feedback = (
            f"ROW COUNT: {n_pred - n_gold} extra row(s) hallucinated. "
            f"Expected {n_gold}, got {n_pred}. Do not invent rows not present in the image."
        )

    # ── Per-row field scoring ──
    row_scores = []
    all_row_issues: list[tuple[int, list[str]]] = []

    for i, gold_row in enumerate(gold_rows):
        if i >= n_pred:
            row_scores.append(0.0)
            all_row_issues.append((i, [f"Row {i+1} completely missing from output."]))
            continue
        row_score, row_issues = _score_row(gold_row, pred_rows[i])
        row_scores.append(row_score)
        if row_issues:
            all_row_issues.append((i, row_issues))

    avg_field_score = sum(row_scores) / len(row_scores)
    final_score = round(0.2 * row_count_score + 0.8 * avg_field_score, 4)

    # ── Plain float path (dspy.Evaluate / inference) ──
    if not pred_name:
        return final_score

    # ── GEPA feedback path ──
    if not all_row_issues and not count_feedback:
        return ScoreWithFeedback(
            score=final_score,
            feedback=f"Perfect extraction across all {n_gold} row(s). Score: {final_score}."
        )

    field_summary = _summarize_issues(all_row_issues)
    feedback_parts = [f"Score: {final_score:.2f}"]
    if count_feedback:
        feedback_parts.append(count_feedback)
    if field_summary:
        feedback_parts.append(field_summary)

    return ScoreWithFeedback(
        score=final_score,
        feedback="\n".join(feedback_parts)
    )