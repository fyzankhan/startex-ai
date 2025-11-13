# semantic_engine.py
# -----------------------------------------------------------------------------
# Lightweight, production-ready semantic engine with the same features:
# 1) overall_sentiment (with intensity & sarcasm), location-aware
# 2) extract_buzzwords (smart=KeyBERT, simple=tokenize), location-aware
# 3) analyze_comment (includes aspect via lightweight KeyBERT-based inference)
# 4) compute_similarities (embedding + sentiment-aware adjustment)
# 5) actionable_recommendations (global + by_location)
# 6) segment_by (re-uses overall_sentiment)
#
# Space-saving choices:
# - No PyABSA (heavy). Aspects via KeyBERT top phrases (multilingual).
# - Smaller models + unified HF cache dir.
# - CPU by default.
# -----------------------------------------------------------------------------

import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT

# -----------------------------------------------------------------------------
# Config (override via env). Keep them small & multilingual.
# -----------------------------------------------------------------------------
SENTIMENT_MODEL = os.getenv(
    "SENTIMENT_MODEL",
    "lxyuan/distilbert-base-multilingual-cased-sentiments-student"  # ~250–300MB
)
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"   # ~120–200MB
)

# Single cache dir to avoid duplicates across libs/containers
CACHE_DIR = os.getenv("HF_CACHE", "/models/hf-cache")
os.environ.setdefault("HF_HOME", CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", CACHE_DIR)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", CACHE_DIR)

# CPU by default (saves GPU blobs and keeps infra simple)
DEVICE = -1

# -----------------------------------------------------------------------------
# Model init
# -----------------------------------------------------------------------------
sentiment_model = pipeline(
    "sentiment-analysis",
    model=SENTIMENT_MODEL,
    device=DEVICE,
    cache_dir=CACHE_DIR,
)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL,
    cache_folder=CACHE_DIR,
)

kw_model = KeyBERT(model=embedding_model)

# Some HF sentiment heads return LABEL_0..2; others return strings. Normalize.
LABEL_MAP: Dict[str, str] = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}

# Minimal stopwords for "simple" buzzword mode (English + a few Urdu/Roman-Urdu)
STOPWORDS = {
    "the", "is", "was", "very", "but", "however", "and", "or", "of", "to",
    "bohot", "tha", "thi", "hai", "hain", "kuch", "acha", "nhi"
}

# -----------------------------------------------------------------------------
# Helpers (I/O normalization + grouping)
# -----------------------------------------------------------------------------
def _normalize_comments(comments: List[Any]) -> List[Dict[str, Any]]:
    """
    Accepts:
      - list[str]
      - list[dict] with 'comment' or 'reason' and optional 'locationID'
    Returns: list of {'comment': str, 'locationID': Any}
    """
    norm: List[Dict[str, Any]] = []
    for item in comments:
        if isinstance(item, str):
            norm.append({"comment": item, "locationID": None})
        elif isinstance(item, dict):
            text = item.get("comment") or item.get("reason") or ""
            loc = item.get("locationID", None)
            norm.append({"comment": text, "locationID": loc})
        else:
            norm.append({"comment": str(item), "locationID": None})
    return norm


def _group_by_location(items: List[Dict[str, Any]]) -> Dict[Any, List[Dict[str, Any]]]:
    groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for it in items:
        groups[it.get("locationID", None)].append(it)
    return groups


# -----------------------------------------------------------------------------
# Core Engine
# -----------------------------------------------------------------------------
class SemanticEngine:

    # -------------------------- Sentiment ------------------------------------
    @staticmethod
    def overall_sentiment(comments: List[Any]) -> dict:
        norm = _normalize_comments(comments)
        if not norm:
            return {"distribution": {}, "total_comments": 0, "classified_comments": [], "by_location": {}}

        classified: List[Dict[str, Any]] = []
        for row in norm:
            comment = row["comment"]
            locationID = row["locationID"]

            if not comment or not str(comment).strip():
                classified.append({
                    "comment": comment,
                    "locationID": locationID,
                    "sentiment": "neutral",
                    "intensity": "low",
                    "sarcasm": False
                })
                continue

            clauses = SemanticEngine.split_comment(comment) or [comment]
            try:
                results = sentiment_model(clauses, truncation=True, max_length=256)
            except Exception:
                results = [{"label": "neutral", "score": 0.0} for _ in clauses]

            mapped = [LABEL_MAP.get(r["label"], str(r["label"])).lower() for r in results]

            sentiment = SemanticEngine._majority_sentiment(mapped)
            intensity = SemanticEngine._emotional_intensity(mapped)
            sarcasm = SemanticEngine._sarcasm_detect(comment, mapped)

            classified.append({
                "comment": comment,
                "locationID": locationID,
                "sentiment": sentiment,
                "intensity": intensity,
                "sarcasm": sarcasm
            })

        counts = Counter(c["sentiment"] for c in classified)
        total = len(classified)
        percentages = {k: round((v / total) * 100, 2) for k, v in counts.items()}

        # per-location distributions
        by_location: Dict[Any, Dict[str, float]] = {}
        loc_groups = _group_by_location(classified)
        for loc, items in loc_groups.items():
            loc_counts = Counter(x["sentiment"] for x in items)
            loc_total = len(items)
            by_location[loc] = {k: round((v / loc_total) * 100, 2) for k, v in loc_counts.items()}

        return {
            "distribution": percentages,
            "total_comments": total,
            "classified_comments": classified,
            "by_location": by_location
        }

    # -------------------------- Buzzwords ------------------------------------
    @staticmethod
    def extract_buzzwords(comments: List[Any], top_n: int = 10, mode: str = "smart") -> dict:
        norm = _normalize_comments(comments)
        if not norm:
            return {"global": {"positive": [], "negative": [], "all": []}, "by_location": {}}

        classified = SemanticEngine.overall_sentiment(norm)["classified_comments"]

        def accumulate(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, int]]]:
            pos_bucket: List[str] = []
            neg_bucket: List[str] = []

            for c in rows:
                text = c.get("comment") or ""
                if not text:
                    continue
                try:
                    if mode == "simple":
                        tokens = SemanticEngine.tokenize(text)
                        if c["sentiment"] == "positive":
                            pos_bucket.extend(tokens)
                        elif c["sentiment"] == "negative":
                            neg_bucket.extend(tokens)
                    else:  # smart = KeyBERT
                        keyphrases = kw_model.extract_keywords(
                            text,
                            keyphrase_ngram_range=(1, 2),
                            stop_words="english",
                            top_n=3
                        )
                        phrases = [kp[0] for kp in keyphrases]
                        if c["sentiment"] == "positive":
                            pos_bucket.extend(phrases)
                        elif c["sentiment"] == "negative":
                            neg_bucket.extend(phrases)
                except Exception:
                    continue

            pos = Counter(pos_bucket).most_common(top_n)
            neg = Counter(neg_bucket).most_common(top_n)
            allc = Counter(pos_bucket + neg_bucket).most_common(top_n)
            return {
                "positive": [{"word": w, "count": n} for w, n in pos],
                "negative": [{"word": w, "count": n} for w, n in neg],
                "all": [{"word": w, "count": n} for w, n in allc],
            }

        global_counts = accumulate(classified)

        by_location: Dict[Any, Dict[str, Any]] = {}
        loc_groups = _group_by_location(classified)
        for loc, rows in loc_groups.items():
            by_location[loc] = accumulate(rows)

        return {"global": global_counts, "by_location": by_location}

    # ----------------------- Single Comment Analysis -------------------------
    @staticmethod
    def analyze_comment(comment: str, locationID: Any = None) -> dict:
        clauses = SemanticEngine.split_comment(comment) or [comment]
        aspects: List[Dict[str, Any]] = []
        try:
            results = sentiment_model(clauses, truncation=True, max_length=256)

            # Lightweight aspect inference:
            # use top KeyBERT phrase(s) from the *full* comment, fallback "general"
            try:
                keyphrases = kw_model.extract_keywords(
                    comment, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=3
                )
                extracted = [kp[0].strip() for kp in keyphrases if str(kp[0]).strip()]
            except Exception:
                extracted = []

            main_aspect = (extracted[0] if extracted else "general")

            for clause, r in zip(clauses, results):
                aspects.append({
                    "clause": clause,
                    "locationID": locationID,
                    "aspect": main_aspect,
                    "label": LABEL_MAP.get(r["label"], str(r["label"])).lower(),
                    "score": float(r["score"])
                })
        except Exception as e:
            # fallback path
            aspects = [{"clause": c, "locationID": locationID, "aspect": "general", "label": "error", "score": 0.0}
                       for c in clauses]

        return {"comment": comment, "locationID": locationID, "aspects": aspects}

    # ---------------- Sentiment-aware Semantic Similarity ---------------------
    @staticmethod
    def compute_similarities(comments: List[Any]) -> List[dict]:
        norm = _normalize_comments(comments)
        if len(norm) < 2:
            return []

        sentiments: List[str] = []
        aspects: List[str] = []
        locs: List[Any] = []
        texts: List[str] = []

        for row in norm:
            c = row["comment"]
            loc = row["locationID"]
            analysis = SemanticEngine.analyze_comment(c, locationID=loc)
            if analysis["aspects"]:
                sentiments.append(analysis["aspects"][0]["label"])
                aspects.append(analysis["aspects"][0]["aspect"])
            else:
                sentiments.append("neutral")
                aspects.append("general")
            locs.append(loc)
            texts.append(c)

        embeddings = embedding_model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        sim_matrix = util.cos_sim(embeddings, embeddings)

        results: List[dict] = []
        n = len(texts)
        for i in range(n):
            for j in range(i + 1, n):
                semantic = float(sim_matrix[i][j].item())
                sentiment_diff = 1.0 if sentiments[i] != sentiments[j] else 0.0
                adjusted = round(max(0.0, semantic - 0.3 * sentiment_diff), 4)
                results.append({
                    "comment1": texts[i],
                    "locationID1": locs[i],
                    "comment2": texts[j],
                    "locationID2": locs[j],
                    "semantic_similarity": round(semantic, 4),
                    "adjusted_similarity": adjusted,
                    "sentiment_pair": f"{sentiments[i]}-{sentiments[j]}",
                    "aspect_pair": f"{aspects[i]}-{aspects[j]}",
                })
        return results

    # ----------------------- Actionable Recommendations -----------------------
    @staticmethod
    def actionable_recommendations(comments: List[Any]) -> dict:
        norm = _normalize_comments(comments)

        def recs_for(texts: List[str]) -> List[str]:
            joined = " ".join(texts).lower()
            recs: List[str] = []
            if any(x in joined for x in ["delay", "slow", "late", "wait", "waiting", "queue"]):
                recs.append("Improve response and processing speed.")
            if any(x in joined for x in ["price", "cost", "expensive", "مہنگ", "مہنگا", "مہنگی", "مہنگے"]):
                recs.append("Re-evaluate pricing and value offering.")
            if any(x in joined for x in ["quality", "fault", "issue", "error", "performance", "bug", "crash"]):
                recs.append("Enhance quality control and reliability.")
            if any(x in joined for x in ["support", "help", "staff", "team", "agent"]):
                recs.append("Invest in staff training and customer support.")
            if any(x in joined for x in ["app", "website", "portal", "interface", "ui", "ux", "system"]):
                recs.append("Improve user experience and system performance.")
            if not recs:
                recs.append("Maintain quality and monitor sentiment regularly.")
            return recs

        global_recs = recs_for([r["comment"] for r in norm if r["comment"]])

        by_location: Dict[Any, List[str]] = {}
        loc_groups = _group_by_location(norm)
        for loc, rows in loc_groups.items():
            by_location[loc] = recs_for([r["comment"] for r in rows if r["comment"]])

        return {"global": global_recs, "by_location": by_location}

    # ----------------------------- Segmentation -------------------------------
    @staticmethod
    def segment_by(comments: List[Dict[str, Any]], field: str) -> dict:
        """
        comments: list of dicts like {brand, product_line, locationID, comment}
        field: e.g., "brand" | "product_line" | "locationID"
        """
        groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        for row in comments:
            key = row.get(field, "Unknown")
            comment = row.get("comment") or row.get("reason")
            if comment:
                groups[key].append({"comment": comment, "locationID": row.get("locationID")})

        segmented: Dict[Any, Dict[str, float]] = {}
        for key, rows in groups.items():
            segmented[key] = SemanticEngine.overall_sentiment(rows)["distribution"]
        return segmented

    # ----------------------------- Utilities ---------------------------------
    @staticmethod
    def split_comment(comment: str) -> List[str]:
        splitters = ["لیکن", "مگر", "but", "however", ".", "!", "?"]
        pattern = "|".join(map(re.escape, splitters))
        return [c.strip() for c in re.split(pattern, comment, flags=re.IGNORECASE) if c.strip()]

    @staticmethod
    def tokenize(text: str) -> List[str]:
        words = re.findall(r"\b\w+\b", text.lower())
        return [w for w in words if w not in STOPWORDS and len(w) > 2]

    @staticmethod
    def _majority_sentiment(labels: List[str]) -> str:
        unique = set(labels)
        if len(unique) == 1:
            return labels[0]
        elif unique == {"neutral", "positive"}:
            return "positive"
        elif unique == {"neutral", "negative"}:
            return "negative"
        elif "positive" in unique and "negative" in unique:
            return "neutral"
        return max(labels, key=labels.count)

    @staticmethod
    def _emotional_intensity(labels: List[str]) -> str:
        strong = labels.count("positive") + labels.count("negative")
        if strong >= len(labels) * 0.7:
            return "high"
        elif strong >= len(labels) * 0.4:
            return "medium"
        return "low"

    @staticmethod
    def _sarcasm_detect(comment: str, labels: List[str]) -> bool:
        text = comment.lower()
        if ("!" in text or "wow" in text or "great" in text) and "negative" in labels:
            return True
        return False
