# semantic_engine.py
# Clean, optimized semantic engine for large comment sets (1–20k)
# Features:
# - Deduplication with count preservation
# - Fast Urdu/Roman confusing-clause detection
# - Lightweight sentiment (lexicon-based)
# - Buzzwords extraction (simple)
# - Actionable insights
# - Segmentation
# - Scales on Azure B-series

import re
from collections import defaultdict, Counter
from typing import Any, Dict, List


# ---------------------------------------------------------------------
# 1) Urdu / Roman Urdu contradiction words
# ---------------------------------------------------------------------
CONTRADICT_WORDS = [
    "lekin", "magar", "mager", "however", "but",
    "لیکن", "مگر"
]

# Small positive & negative lexicons (fast, ML-free)
POS_WORDS = {
    "acha", "achi", "achay", "theek", "bohot acha", "satisfied",
    "khush", "great", "amazing", "excellent"
}

NEG_WORDS = {
    "bura", "bori", "ganda", "rude", "slow", "wait", "late",
    "issue", "masla", "problem", "bad", "terrible", "poor"
}

STOPWORDS = {
    "the","is","was","very","and","or","of","to","hai","hain",
    "bohot","but","however","lekin","magar"
}


# ---------------------------------------------------------------------
# 2) Normalize comments
# ---------------------------------------------------------------------
def _normalize(comments: List[Any]) -> List[Dict[str, Any]]:
    out = []
    for x in comments:
        if isinstance(x, str):
            out.append({"comment": x, "locationID": None, "count": 1})
        elif isinstance(x, dict):
            txt = x.get("comment") or x.get("reason") or ""
            out.append({
                "comment": txt,
                "locationID": x.get("locationID"),
                "count": 1
            })
        else:
            out.append({"comment": str(x), "locationID": None, "count": 1})
    return out


# ---------------------------------------------------------------------
# 3) Dedupe identical comments (preserve count)
# ---------------------------------------------------------------------
def dedupe_comments(norm_list):
    bucket = {}
    for row in norm_list:
        key = (row["comment"].strip().lower(), row["locationID"])
        if key not in bucket:
            bucket[key] = {
                "comment": row["comment"],
                "locationID": row["locationID"],
                "count": 1
            }
        else:
            bucket[key]["count"] += 1
    return list(bucket.values())


# ---------------------------------------------------------------------
# 4) Clause splitter
# ---------------------------------------------------------------------
def split_comment(text: str) -> List[str]:
    if not text:
        return []
    pattern = "|".join(map(re.escape, CONTRADICT_WORDS))
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------
# 5) Lightweight sentiment (lexicon-based)
# ---------------------------------------------------------------------
def clause_sentiment(text: str) -> str:
    t = text.lower()
    pos = any(w in t for w in POS_WORDS)
    neg = any(w in t for w in NEG_WORDS)
    if pos and not neg:
        return "positive"
    if neg and not pos:
        return "negative"
    return "neutral"


def detect_confusion(text: str) -> List[str]:
    """Return only the contradictory/Urdu confusing clause(s)."""
    lower = text.lower()

    if not any(word in lower for word in CONTRADICT_WORDS):
        return []

    clauses = split_comment(text)
    if len(clauses) < 2:
        return []

    sentiments = [clause_sentiment(c) for c in clauses]
    results = []

    # Compare clause pairs
    for i in range(len(clauses) - 1):
        s1 = sentiments[i]
        s2 = sentiments[i + 1]
        if s1 != s2:
            # Restore the contradiction word before second clause
            for w in CONTRADICT_WORDS:
                if w in lower:
                    results.append(f"{w} {clauses[i+1]}")
                    break

    return results


# ---------------------------------------------------------------------
# 6) Buzzwords (simple token frequency)
# ---------------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    words = re.findall(r"\b\w+\b", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


# ---------------------------------------------------------------------
# 7) Main Engine (sentiment, confusion, buzzwords, segmentation)
# ---------------------------------------------------------------------
class SemanticEngine:

    @staticmethod
    def overall_sentiment(comments: List[Any]) -> dict:
        norm = dedupe_comments(_normalize(comments))
        classified = []

        for row in norm:
            text = row["comment"]
            loc = row["locationID"]
            cnt = row["count"]

            clauses = split_comment(text) or [text]
            sentiments = [clause_sentiment(c) for c in clauses]

            if all(s == "positive" for s in sentiments):
                final = "positive"
            elif all(s == "negative" for s in sentiments):
                final = "negative"
            else:
                final = "neutral"

            classified.append({
                "comment": text,
                "locationID": loc,
                "sentiment": final,
                "count": cnt,
                "confusing_parts": detect_confusion(text)
            })

        # Global distribution
        total = sum(c["count"] for c in classified)
        dist = Counter()
        for c in classified:
            dist[c["sentiment"]] += c["count"]

        distribution = {k: round((v / total) * 100, 2) for k, v in dist.items()}

        # Per-location distribution
        by_loc = {}
        grouped = defaultdict(list)
        for c in classified:
            grouped[c["locationID"]].append(c)

        for loc, rows in grouped.items():
            lc = Counter()
            total_l = 0
            for r in rows:
                lc[r["sentiment"]] += r["count"]
                total_l += r["count"]
            by_loc[loc] = {k: round((v / total_l) * 100, 2) for k, v in lc.items()}

        return {
            "total_comments": total,
            "classified_comments": classified,
            "distribution": distribution,
            "by_location": by_loc
        }

    @staticmethod
    def extract_buzzwords(comments: List[Any], top_n: int = 10) -> dict:
        norm = dedupe_comments(_normalize(comments))
        pos_bucket = []
        neg_bucket = []

        for r in norm:
            t = r["comment"]
            cnt = r["count"]
            sent = clause_sentiment(t)

            tokens = tokenize(t)
            if sent == "positive":
                pos_bucket.extend(tokens * cnt)
            elif sent == "negative":
                neg_bucket.extend(tokens * cnt)

        pos = Counter(pos_bucket).most_common(top_n)
        neg = Counter(neg_bucket).most_common(top_n)
        all_words = Counter(pos_bucket + neg_bucket).most_common(top_n)

        return {
            "global": {
                "positive": [{"word": w, "count": n} for w, n in pos],
                "negative": [{"word": w, "count": n} for w, n in neg],
                "all": [{"word": w, "count": n} for w, n in all_words]
            }
        }

    @staticmethod
    def actionable_recommendations(comments: List[Any]) -> dict:
        norm = dedupe_comments(_normalize(comments))
        joined = " ".join(r["comment"].lower() for r in norm)

        recs = []
        if any(x in joined for x in ["delay", "slow", "late", "wait"]):
            recs.append("Improve response speed.")
        if any(x in joined for x in ["price", "cost", "expensive"]):
            recs.append("Review pricing and value offering.")
        if any(x in joined for x in ["quality", "issue", "error", "bug"]):
            recs.append("Enhance system reliability.")
        if any(x in joined for x in ["support", "staff", "agent"]):
            recs.append("Improve staff/customer support.")
        if any(x in joined for x in ["app", "website", "portal", "system", "ui", "ux"]):
            recs.append("Improve user experience.")
        if not recs:
            recs.append("Monitor sentiment regularly.")

        return {"global": recs}

    @staticmethod
    def segment_by(comments: List[Dict[str, Any]], field: str) -> dict:
        grouped = defaultdict(list)
        for r in comments:
            k = r.get(field, "Unknown")
            if r.get("comment"):
                grouped[k].append({"comment": r["comment"], "locationID": r.get("locationID")})

        result = {}
        for key, items in grouped.items():
            result[key] = SemanticEngine.overall_sentiment(items)["distribution"]

        return result
