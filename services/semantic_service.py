from datetime import date, datetime
from sqlalchemy import text
from services.semantic_engine import SemanticEngine
from config import engine
from typing import Any, Dict, List, Tuple
import os

MAX_SIMILARITY_COMMENTS = int(os.getenv("SEM_MAX_SIM_COMMENTS", "200"))  # cap O(n^2)

class SemanticService:
    """
    Fetch comments from MySQL using filters.
    Run full analysis with SemanticEngine (location-aware).
    No sampling: processes ALL comments returned by the query (except similarity cap).
    """

    @staticmethod
    def _iso(v):
        if v is None:
            return None
        if isinstance(v, (datetime, date)):
            return v.isoformat()
        return str(v)

    @staticmethod
    def _bind_in_clause(column, values, param_prefix, params: dict) -> str:
        """
        Builds a deterministic IN(...) clause with named parameters.
        If values is an explicit empty list, return '1=0' to avoid unbounded queries.
        """
        if isinstance(values, (list, tuple, set)):
            values = [v for v in values if v is not None]
            if len(values) == 0:
                return "1=0"
        else:
            values = [values] if values is not None else []

        if not values:
            return None

        placeholders = []
        for idx, val in enumerate(values):
            key = f"{param_prefix}_{idx}"
            placeholders.append(f":{key}")
            params[key] = val
        return f"{column} IN ({', '.join(placeholders)})"

    @staticmethod
    def get_insights(filters: dict):
        formatID   = filters.get("formatID")
        sectionID  = filters.get("sectionID")
        Tag        = filters.get("Tag")
        locationID = filters.get("locationID")
        dateFrom   = filters.get("dateFrom")
        dateTo     = filters.get("dateTo")
        buzz_mode  = filters.get("mode", "smart")

        # ---------- 1) WHERE clause ----------
        conditions: List[str] = []
        params: Dict[str, Any] = {}

        # formatID
        if formatID is not None:
            if isinstance(formatID, (list, tuple, set)):
                clause = SemanticService._bind_in_clause("formatID", formatID, "formatID", params)
                if clause: conditions.append(clause)
            else:
                conditions.append("formatID = :formatID")
                params["formatID"] = formatID

        # sectionID
        if sectionID is not None:
            clause = SemanticService._bind_in_clause("sectionID", sectionID, "sectionID", params)
            if clause: conditions.append(clause)

        # locationID
        if locationID is not None:
            clause = SemanticService._bind_in_clause("locationID", locationID, "locationID", params)
            if clause: conditions.append(clause)

        # Tag
        if Tag is not None:
            clause = SemanticService._bind_in_clause("Tag", Tag, "Tag", params)
            if clause: conditions.append(clause)

        # Dates
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

        # Non-empty comments only
        conditions.append("reason IS NOT NULL")
        conditions.append("reason != ''")

        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

        # ---------- 2) Query DB ----------
        sql = f"""
        SELECT id, formatID, sectionID, locationID, reason, Date
        FROM responsend
        {where_clause}
        ORDER BY Date DESC
        """

        print("[SQL]", sql, params)

        with engine.connect() as conn:
            rows = conn.execute(text(sql), params).mappings().all()

        if not rows:
            return {
                "filters": filters,
                "total_comments": 0,
                "items": [],
                "analytics": {
                    "sentiment": {"distribution": {}, "total_comments": 0, "classified_comments": [], "by_location": {}},
                    "buzzwords": {"global": {"positive": [], "negative": [], "all": []}, "by_location": {}},
                    "similarities": [],
                    "insights": {"global": [], "by_location": {}},
                    "segmentation": {"by_location": {}, "by_section": {}, "by_format": {}},
                },
                "message": "No comments found",
            }

        # ---------- 3) Build payload ----------
        comments_payload: List[Dict[str, Any]] = []
        for m in rows:
            reason = (m.get("reason") or "").strip()
            if not reason:
                continue
            comments_payload.append({
                "comment": reason,
                "locationID": m.get("locationID"),
                "id": m.get("id"),
                "formatID": m.get("formatID"),
                "sectionID": m.get("sectionID"),
                "date": SemanticService._iso(m.get("Date")),  # <-- JSON-safe
            })

        if not comments_payload:
            return {
                "filters": filters,
                "total_comments": 0,
                "items": [],
                "analytics": {
                    "sentiment": {"distribution": {}, "total_comments": 0, "classified_comments": [], "by_location": {}},
                    "buzzwords": {"global": {"positive": [], "negative": [], "all": []}, "by_location": {}},
                    "similarities": [],
                    "insights": {"global": [], "by_location": {}},
                    "segmentation": {"by_location": {}, "by_section": {}, "by_format": {}},
                },
                "message": "No non-empty comments after filtering",
            }

        # ---------- 4) Run NLP Analysis ----------
        sentiment    = SemanticEngine.overall_sentiment(comments_payload)
        buzzwords    = SemanticEngine.extract_buzzwords(comments_payload, top_n=10, mode=buzz_mode)

        # Guard similarities (O(n^2)). Skip if too many comments or torch missing.
        similarities: List[Dict[str, Any]] = []
        try:
            if len(comments_payload) <= MAX_SIMILARITY_COMMENTS:
                similarities = SemanticEngine.compute_similarities(comments_payload)
            else:
                similarities = []  # or sample first N; keep API stable
        except Exception as _e:
            # log upstream; keep API stable
            similarities = []

        insights = SemanticEngine.actionable_recommendations(comments_payload)

        # Segmentation (re-uses sentiment)
        segmentation = {
            "by_location": SemanticEngine.segment_by(
                [{"comment": r["comment"], "locationID": r["locationID"]} for r in comments_payload],
                "locationID"
            ),
            "by_section": SemanticEngine.segment_by(
                [{"comment": r["comment"], "locationID": r["locationID"], "sectionID": r["sectionID"]} for r in comments_payload],
                "sectionID"
            ),
            "by_format": SemanticEngine.segment_by(
                [{"comment": r["comment"], "locationID": r["locationID"], "formatID": r["formatID"]} for r in comments_payload],
                "formatID"
            ),
        }

        return {
            "filters": filters,
            "total_comments": len(comments_payload),
            "items": comments_payload,  # JSON-safe now
            "analytics": {
                "sentiment": sentiment,
                "buzzwords": buzzwords,
                "similarities": similarities,
                "insights": insights,
                "segmentation": segmentation,
            },
        }
