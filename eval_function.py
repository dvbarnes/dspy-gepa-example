import dspy
from typing import Any
from datetime import datetime
import re
from rapidfuzz import fuzz


def table_extraction_metric(example: dspy.Example, pred: Any, trace=None) -> float:
    """
    GEVAL-style metric for image table extraction.
    Evaluates predicted rows against gold rows across all fields.
    
    Args:
        example: dspy.Example with 'image_path' and 'rows' (gold)
        pred: prediction object with 'rows' attribute
        trace: optional trace for MIPRO compatibility
    
    Returns:
        float score in [0.0, 1.0]
    """
    gold_rows = example.rows if hasattr(example, "rows") else []
    pred_rows = pred.rows if hasattr(pred, "rows") else []

    if not gold_rows and not pred_rows:
        return 1.0
    if not gold_rows or not pred_rows:
        return 0.0

    # --- Row count penalty ---
    row_count_score = 1.0 - abs(len(gold_rows) - len(pred_rows)) / max(len(gold_rows), len(pred_rows))

    # --- Field-level scoring across aligned rows ---
    field_scores = []
    for i, gold_row in enumerate(gold_rows):
        if i >= len(pred_rows):
            field_scores.append(0.0)
            continue
        pred_row = pred_rows[i]
        row_score = _score_row(gold_row, pred_row)
        field_scores.append(row_score)

    avg_field_score = sum(field_scores) / len(field_scores) if field_scores else 0.0

    # Weighted combination
    final_score = 0.2 * row_count_score + 0.8 * avg_field_score
    return round(final_score, 4)


def _score_row(gold: dict, pred: dict) -> float:
    """Score a single predicted row against the gold row."""
    scores = []

    # Date field
    if "date" in gold:
        scores.append(_score_date(gold.get("date"), pred.get("date")))

    # Description field (fuzzy string match)
    if "description" in gold:
        scores.append(_score_text(gold.get("description"), pred.get("description")))

    # Location field (case-insensitive exact / partial)
    if "location" in gold:
        scores.append(_score_text(gold.get("location"), pred.get("location"), weight_partial=True))

    # Amount field (numeric tolerance)
    if "amount" in gold:
        scores.append(_score_numeric(gold.get("amount"), pred.get("amount")))

    return sum(scores) / len(scores) if scores else 0.0


def _score_date(gold_val: Any, pred_val: Any) -> float:
    """Accept exact match or common date format variants."""
    if gold_val is None and pred_val is None:
        return 1.0
    if gold_val is None or pred_val is None:
        return 0.0

    gold_str = str(gold_val).strip()
    pred_str = str(pred_val).strip()

    if gold_str == pred_str:
        return 1.0

    # Normalize to YYYY-MM-DD and compare
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            gold_dt = datetime.strptime(gold_str, fmt)
            pred_dt = datetime.strptime(pred_str, fmt)
            return 1.0 if gold_dt == pred_dt else 0.0
        except ValueError:
            continue

    return 0.0


def _score_text(gold_val: Any, pred_val: Any, weight_partial: bool = False) -> float:
    """RapidFuzz-based similarity for text fields."""
    if gold_val is None and pred_val is None:
        return 1.0
    if gold_val is None or pred_val is None:
        return 0.0

    gold_str = str(gold_val).strip()
    pred_str = str(pred_val).strip()

    if weight_partial:
        # partial_ratio handles substring cases like "Mountain View" vs "Mountain"
        score = fuzz.partial_ratio(gold_str.lower(), pred_str.lower()) / 100.0
    else:
        # token_sort_ratio handles word-order variance (good for descriptions)
        score = fuzz.token_sort_ratio(gold_str.lower(), pred_str.lower()) / 100.0

    return round(score, 4)

def _score_numeric(gold_val: Any, pred_val: Any, tolerance: float = 0.01) -> float:
    """Exact match with a small float tolerance for amounts."""
    if gold_val is None and pred_val is None:
        return 1.0
    if gold_val is None or pred_val is None:
        return 0.0
    try:
        g = float(gold_val)
        p = float(pred_val)
        if g == 0 and p == 0:
            return 1.0
        if g == 0:
            return 0.0
        return 1.0 if abs(g - p) / abs(g) <= tolerance else 0.0
    except (ValueError, TypeError):
        return 1.0 if str(gold_val).strip() == str(pred_val).strip() else 0.0