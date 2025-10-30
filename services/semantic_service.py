from services.semantic_engine import SemanticEngine
from sqlalchemy import text
from config import engine

class SemanticService:
    """
    Fetch comments from MySQL using filters.
    Run full analysis with SemanticEngine (location-aware).
    No sampling: processes ALL comments returned by the query.
    """

    @staticmethod
    def get_insights(filters: dict):
        formatID   = filters.get("formatID")
        sectionID  = filters.get("sectionID")
        Tag        = filters.get("Tag")
        locationID = filters.get("locationID")
        dateFrom   = filters.get("dateFrom")
        dateTo     = filters.get("dateTo")

        # ---------- 1) WHERE clause ----------
        conditions = []
        params = {}

        def bind_in_clause(column, values, param_prefix):
            values = list(values) if isinstance(values, (list, tuple, set)) else [values]
            values = [v for v in values if v is not None]
            if not values:
                return None
            placeholders = []
            for idx, val in enumerate(values):
                key = f"{param_prefix}_{idx}"
                placeholders.append(f":{key}")
                params[key] = val
            return f"{column} IN ({', '.join(placeholders)})"

        if formatID is not None:
            if isinstance(formatID, (list, tuple, set)):
                clause = bind_in_clause("formatID", formatID, "formatID")
                if clause: conditions.append(clause)
            else:
                conditions.append("formatID = :formatID")
                params["formatID"] = formatID

        if sectionID is not None:
            if isinstance(sectionID, (list, tuple, set)):
                clause = bind_in_clause("sectionID", sectionID, "sectionID")
                if clause: conditions.append(clause)
            else:
                conditions.append("sectionID = :sectionID")
                params["sectionID"] = sectionID  # fixed key (no trailing space)

        if locationID is not None:
            if isinstance(locationID, (list, tuple, set)):
                clause = bind_in_clause("locationID", locationID, "locationID")
                if clause: conditions.append(clause)
            else:
                conditions.append("locationID = :locationID")
                params["locationID"] = locationID

        if Tag is not None:
            if isinstance(Tag, (list, tuple, set)):
                clause = bind_in_clause("Tag", Tag, "Tag")
                if clause: conditions.append(clause)
            else:
                conditions.append("Tag = :Tag")
                params["Tag"] = Tag

        # ---------- 2) Date filter ----------
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

        # ---------- 3) Query DB ----------
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
            return {"message": "No comments found", "filters": filters}

        comments_data = [dict(row) for row in rows]

        # Build location-aware payload for the engine (ALL comments; no sampling)
        comments_payload = [
            {
                "comment": r.get("reason") or "",
                "locationID": r.get("locationID"),
                "id": r.get("id"),
                "formatID": r.get("formatID"),
                "sectionID": r.get("sectionID"),
                "date": r.get("Date"),
            }
            for r in comments_data
            if (r.get("reason") or "").strip()
        ]

        if not comments_payload:
            return {"message": "No non-empty comments after filtering", "filters": filters}

        # ---------- 4) Run NLP Analysis (ALL comments) ----------
        sentiment     = SemanticEngine.overall_sentiment(comments_payload)          # distribution + by_location
        buzzwords     = SemanticEngine.extract_buzzwords(comments_payload, top_n=10)# global + by_location
        similarities  = SemanticEngine.compute_similarities(comments_payload)       # includes locationID1/2 (ALL pairs)
        insights      = SemanticEngine.actionable_recommendations(comments_payload) # global + by_location

        # Segmentation by fields we actually have
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

        # ---------- 5) Response ----------
        return {
            "filters": filters,
            "total_comments": len(comments_payload),
            "items": comments_payload,  # raw items (ALL)
            "analytics": {
                "sentiment": sentiment,          # includes by_location + per-comment list
                "buzzwords": buzzwords,          # global + by_location
                "similarities": similarities,    # ALL pairwise (may be large)
                "insights": insights,            # global + by_location
                "segmentation": segmentation,    # by_location/section/format
            },
        }
