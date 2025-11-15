# semantic_service.py
# Clean, fast, production-ready service layer for semantic analytics

from datetime import date, datetime
from typing import Any, Dict, List

from sqlalchemy import text
from config import engine

from services.semantic_engine import SemanticEngine


MAX_SIMILARITY_COMMENTS = 0  # disabled (engine S2 does not use embeddings)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _iso(v):
    if v is None:
        return None
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    return str(v)


def _bind_in_clause(column, values, prefix, params: dict) -> str:
    if isinstance(values, (list, tuple, set)):
        values = [v for v in values if v is not None]
        if len(values) == 0:
            return "1=0"
    else:
        values = [values] if values is not None else []

    if not values:
        return None

    ph = []
    for i, val in enumerate(values):
        key = f"{prefix}_{i}"
        ph.append(f":{key}")
        params[key] = val
    return f"{column} IN ({', '.join(ph)})"


# ---------------------------------------------------------------------
# Main Service
# ---------------------------------------------------------------------
class SemanticService:
    """
    Fetches comments, applies filters, returns:
    - sentiment distribution (deduped)
    - buzzwords (deduped)
    - confusion detection (built-in engine)
    - recommendations
    - segmentation
    """

    @staticmethod
    def get_insights(filters: dict):
        formatID   = filters.get("formatID")
        sectionID  = filters.get("sectionID")
        Tag        = filters.get("Tag")
        locationID = filters.get("locationID")
        dateFrom   = filters.get("dateFrom")
        dateTo     = filters.get("dateTo")
        mode       = filters.get("mode", "smart")

        conditions = []
        params: Dict[str, Any] = {}

        # formatID
        if formatID is not None:
            clause = _bind_in_clause("formatID", formatID, "formatID", params)
            if clause:
                conditions.append(clause)

        # sectionID
        if sectionID is not None:
            clause = _bind_in_clause("sectionID", sectionID, "sectionID", params)
            if clause:
                conditions.append(clause)

        # locationID
        if locationID is not None:
            clause = _bind_in_clause("locationID", locationID, "locationID", params)
            if clause:
                conditions.append(clause)

        # Tag
        if Tag is not None:
            clause = _bind_in_clause("Tag", Tag, "Tag", params)
            if clause:
                conditions.append(clause)

        # date filters
        if dateFrom and dateTo:
            conditions.append("DATE(Date) BETWEEN :dateFrom AND :dateTo")
            params["dateFrom"] = dateFrom
            params["dateTo"]   = dateTo
        elif dateFrom:
            conditions.append("DATE(Date) >= :dateFrom")
            params["dateFrom"] = dateFrom
        elif dateTo:
            conditions.append("DATE(Date) <= :dateTo")
            params["dateTo"] = dateTo

        # Only non-empty comments
        conditions.append("reason IS NOT NULL")
        conditions.append("reason != ''")

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        sql = f"""
        SELECT id, formatID, sectionID, locationID, reason, Date
        FROM responsend
        {where_clause}
        ORDER BY Date DESC
        """

        with engine.connect() as conn:
            rows = conn.execute(text(sql), params).mappings().all()

        if not rows:
            return {
                "filters": filters,
                "total_comments": 0,
                "items": [],
                "analytics": {
                    "sentiment": {
                        "distribution": {},
                        "total_comments": 0,
                        "classified_comments": [],
                        "by_location": {}
                    },
                    "buzzwords": {
                        "global": {"positive": [], "negative": [], "all": []}
                    },
                    "insights": {"global": [], "by_location": {}},
                    "segmentation": {
                        "by_location": {},
                        "by_section": {},
                        "by_format": {}
                    }
                },
                "message": "No comments found"
            }

        # Prepare payload (UI-compatible)
        items = []
        for r in rows:
            text_val = (r.get("reason") or "").strip()
            if not text_val:
                continue

            items.append({
                "id": r.get("id"),
                "comment": text_val,
                "locationID": r.get("locationID"),
                "formatID": r.get("formatID"),
                "sectionID": r.get("sectionID"),
                "date": _iso(r.get("Date")),
            })

        if not items:
            return {
                "filters": filters,
                "total_comments": 0,
                "items": [],
                "analytics": {},
                "message": "No non-empty comments after filtering"
            }

        # NLP Processing
        sentiment = SemanticEngine.overall_sentiment(items)
        buzzwords = SemanticEngine.extract_buzzwords(items, top_n=10)
        insights  = SemanticEngine.actionable_recommendations(items)

        # segmentation
        segmentation = {
            "by_location": SemanticEngine.segment_by(items, "locationID"),
            "by_section": SemanticEngine.segment_by(items, "sectionID"),
            "by_format":  SemanticEngine.segment_by(items, "formatID"),
        }

        return {
            "filters": filters,
            "total_comments": len(items),
            "items": items,
            "analytics": {
                "sentiment": sentiment,
                "buzzwords": buzzwords,
                "insights": insights,
                "segmentation": segmentation,
            },
        }
