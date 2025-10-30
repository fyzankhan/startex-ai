import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT

# ---------- Robust PyABSA import (v2 first, then v1; graceful fallback) ----------
import pyabsa
from packaging import version
PYABSA_VER = version.parse(getattr(pyabsa, "__version__", "0.0.0"))

_ASPECT_EXTRACTOR = None
def _load_aspect_extractor():
    """
    Singleton aspect extractor compatible with PyABSA v2.x and v1.x.
    Returns None if loading fails; engine will fall back to 'general' aspect.
    """
    global _ASPECT_EXTRACTOR
    if _ASPECT_EXTRACTOR is not None:
        return _ASPECT_EXTRACTOR

    checkpoint = os.getenv("PYABSA_CHECKPOINT", "multilingual")

    try:
        if PYABSA_VER >= version.parse("2.0.0"):
            from pyabsa import ATEPCCheckpointManager
            _ASPECT_EXTRACTOR = ATEPCCheckpointManager.get_aspect_extractor(
                checkpoint=checkpoint,
                auto_device=True,
            )
        else:
            from pyabsa import AspectExtractor as V1AspectExtractor
            _ASPECT_EXTRACTOR = V1AspectExtractor.from_pretrained(
                checkpoint,
                auto_device=True,
            )
    except Exception as e:
        print(f"[WARN] PyABSA extractor failed to load: {e}\n"
              f"       Engine will use 'general' for aspect.\n"
              f"       If checkpoint errors persist, try: pip install 'transformers<=4.29.0' -U")
        _ASPECT_EXTRACTOR = None

    return _ASPECT_EXTRACTOR


# ============================================================
# Model Setup
# ============================================================
SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "cardiffnlp/twitter-xlm-roberta-base-sentiment")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

DEVICE = 0 if torch.cuda.is_available() else -1

sentiment_model = pipeline("sentiment-analysis", model=SENTIMENT_MODEL, device=DEVICE)
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
kw_model = KeyBERT(model=embedding_model)

LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral"
}

STOPWORDS = {
    "the", "is", "was", "very", "but", "however", "and", "or", "of", "to",
    "bohot", "tha", "thi", "hai", "hain", "kuch", "acha", "nhi"
}


# ============================================================
# Helpers (location-aware normalization)
# ============================================================
def _normalize_comments(
    comments: List[Any]
) -> List[Dict[str, Any]]:
    """
    Accepts:
      - list[str]
      - list[dict] with keys like 'comment' (or 'reason') and optional 'locationID'/'locationID'
    Returns list of dicts: {'comment': str, 'locationID': Any}
    """
    norm: List[Dict[str, Any]] = []
    for item in comments:
        if isinstance(item, str):
            norm.append({"comment": item, "locationID": None})
        elif isinstance(item, dict):
            text = item.get("comment") or item.get("reason") or ""
            # tolerate both camel and snake case
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


# ============================================================
# Semantic Engine
# ============================================================
class SemanticEngine:

    # ---------- 1. Sentiment ----------
    @staticmethod
    def overall_sentiment(comments: List[Any]) -> dict:
        norm = _normalize_comments(comments)
        if not norm:
            return {"distribution": {}, "total_comments": 0, "classified_comments": [], "by_location": {}}

        classified = []
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

            clauses = SemanticEngine.split_comment(comment)
            if not clauses:
                clauses = [comment]

            try:
                results = sentiment_model(clauses, truncation=True, max_length=256)
            except Exception:
                results = [{"label": "neutral", "score": 0.0} for _ in clauses]

            mapped = [LABEL_MAP.get(r["label"], r["label"]).lower() for r in results]

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

        counts = Counter([c["sentiment"] for c in classified])
        total = len(classified)
        percentages = {k: round((v / total) * 100, 2) for k, v in counts.items()}

        # per-location distributions
        by_location: Dict[Any, Dict[str, float]] = {}
        loc_groups = _group_by_location(classified)
        for loc, items in loc_groups.items():
            loc_counts = Counter([x["sentiment"] for x in items])
            loc_total = len(items)
            by_location[loc] = {k: round((v / loc_total) * 100, 2) for k, v in loc_counts.items()}

        return {
            "distribution": percentages,
            "total_comments": total,
            "classified_comments": classified,
            "by_location": by_location
        }

    # ---------- 2. Buzzwords ----------
    @staticmethod
    def extract_buzzwords(comments: List[Any], top_n: int = 10, mode: str = "smart") -> dict:
        norm = _normalize_comments(comments)
        if not norm:
            return {
                "global": {"positive": [], "negative": [], "all": []},
                "by_location": {}
            }

        # reuse sentiment classification (location-aware)
        classified = SemanticEngine.overall_sentiment(norm)["classified_comments"]

        def accumulate(rows: List[Dict[str, Any]]) -> Dict[str, List[Tuple[str, int]]]:
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
                    else:
                        keyphrases = kw_model.extract_keywords(
                            text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=3
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

        # global
        global_counts = accumulate(classified)

        # by location
        by_location: Dict[Any, Dict[str, Any]] = {}
        loc_groups = _group_by_location(classified)
        for loc, rows in loc_groups.items():
            by_location[loc] = accumulate(rows)

        return {
            "global": global_counts,
            "by_location": by_location
        }

    # ---------- 3. Single Comment Analysis ----------
    @staticmethod
    def analyze_comment(comment: str, locationID: Any = None) -> dict:
        clauses = SemanticEngine.split_comment(comment)
        if not clauses:
            clauses = [comment]

        aspects = []
        try:
            results = sentiment_model(clauses, truncation=True, max_length=256)

            extractor = _load_aspect_extractor()
            extracted = ["general"]
            if extractor is not None:
                absa_out = extractor.extract_aspect(
                    inference_source=[comment],
                    pred_sentiment=False
                )
                if absa_out and isinstance(absa_out, list):
                    first = absa_out[0]
                    cand = first.get("aspect") or first.get("aspects")
                    if cand:
                        if isinstance(cand, str):
                            extracted = [cand]
                        elif isinstance(cand, (list, tuple)):
                            extracted = [str(x) for x in cand if str(x).strip()]

            main_aspect = (extracted[0] if extracted else "general").strip() or "general"

            for clause, r in zip(clauses, results):
                aspects.append({
                    "clause": clause,
                    "locationID": locationID,
                    "aspect": main_aspect,
                    "label": LABEL_MAP.get(r["label"], r["label"]).lower(),
                    "score": float(r["score"])
                })
        except Exception as e:
            print(f"[WARN] analyze_comment fallback due to error: {e}")
            aspects = [{"clause": c, "locationID": locationID, "aspect": "general", "label": "error", "score": 0.0} for c in clauses]

        return {"comment": comment, "locationID": locationID, "aspects": aspects}

    # ---------- 4. Semantic Similarity (Sentiment-Aware) ----------
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

        results = []
        n = len(texts)
        for i in range(n):
            for j in range(i + 1, n):
                semantic = sim_matrix[i][j].item()
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
                    "aspect_pair": f"{aspects[i]}-{aspects[j]}"
                })
        return results

    # ---------- 5. Actionable Insights ----------
    @staticmethod
    def actionable_recommendations(comments: List[Any]) -> dict:
        norm = _normalize_comments(comments)

        def recs_for(texts: List[str]) -> List[str]:
            joined = " ".join(texts).lower()
            recs: List[str] = []
            if any(x in joined for x in ["delay", "slow", "late, wait", "waiting", "queue"]):
                recs.append("Improve response and processing speed.")
            if any(x in joined for x in ["price", "cost", "expensive", "مہنگ"]):
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

        # global
        global_recs = recs_for([r["comment"] for r in norm if r["comment"]])

        # per location
        by_location: Dict[Any, List[str]] = {}
        loc_groups = _group_by_location(norm)
        for loc, rows in loc_groups.items():
            by_location[loc] = recs_for([r["comment"] for r in rows if r["comment"]])

        return {"global": global_recs, "by_location": by_location}

    # ---------- 6. Segmentation ----------
    @staticmethod
    def segment_by(comments: List[Dict[str, Any]], field: str) -> dict:
        """
        comments: list of dicts like {brand, product_line, locationID, comment}
        field: any top-level key to segment by, e.g. "brand" | "product_line" | "locationID"
        """
        groups = defaultdict(list)
        for row in comments:
            key = row.get(field, "Unknown")
            comment = row.get("comment") or row.get("reason")
            if comment:
                groups[key].append({"comment": comment, "locationID": row.get("locationID")})

        segmented = {}
        for key, rows in groups.items():
            segmented[key] = SemanticEngine.overall_sentiment(rows)["distribution"]
        return segmented

    # ---------- Helpers ----------
    @staticmethod
    def split_comment(comment: str) -> List[str]:
        splitters = ["لیکن", "مگر", "but", "however", ".", "!", "?"]
        pattern = "|".join(map(re.escape, splitters))
        return [c.strip() for c in re.split(pattern, comment, flags=re.IGNORECASE) if c.strip()]

    @staticmethod
    def tokenize(text: str) -> List[str]:
        words = re.findall(r"\b\w+\b", text.lower())
        return [w for w in words if w not in STOPWORDS and len(w) > 2]

    # ---------- Internal Add-ons ----------
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
