from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError, TimeoutError as SQLATimeout

from config import engine
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


def _normalize_branch(name: Optional[str]) -> str:
    if not name:
        return ""
    s = name.lower()
    s = s.replace("–", " ").replace("—", " ").replace("-", " ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _detect_date_range_from_scope(scope: str) -> Optional[Tuple[str, str]]:
    scope = (scope or "").lower().strip()
    today = date.today()
    start: Optional[date] = None
    end: Optional[date] = None
    if scope in ("", "all", "any"):
        return None
    if scope == "today":
        start = today
        end = today
    elif scope == "yesterday":
        start = today - timedelta(days=1)
        end = today - timedelta(days=1)
    elif scope in ("last_week", "last week"):
        start = today - timedelta(days=7)
        end = today
    elif scope in ("this_month", "this month"):
        start = today.replace(day=1)
        if start.month == 12:
            end = date(start.year, 12, 31)
        else:
            nm = date(start.year, start.month + 1, 1)
            end = nm - timedelta(days=1)
    elif scope in ("last_30_days", "last 30 days"):
        start = today - timedelta(days=30)
        end = today
    if start is None or end is None:
        return None
    return f"{start.isoformat()} 00:00:00", f"{end.isoformat()} 23:59:59"


def _parse_explicit_date(question: str) -> Optional[Tuple[str, str]]:
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", question)
    if not m:
        return None
    try:
        d = datetime.strptime(m.group(0), "%Y-%m-%d").date()
    except ValueError:
        return None
    return f"{d.isoformat()} 00:00:00", f"{d.isoformat()} 23:59:59"


INTENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an intent parser for a customer feedback analytics chatbot.\n"
                "You MUST reply with STRICT JSON, no extra text.\n\n"
                "Extract these fields:\n"
                "- query_type: one of ['nps_metric','branch_analytics','date_analytics','topic_analytics','sentiment_analytics','customer_trace','compare','examples','insight','complex_reasoning','mixed']\n"
                "- branch: primary branch/area string if mentioned, else null\n"
                "- branch_b: secondary branch string for comparisons, else null\n"
                "- customer: customer identifier string if user refers to a specific customer (e.g. 'customer 0'), else null\n"
                "- sentiment: one of ['negative','positive','all'] (infer from words like complaint/issue/problem vs happy/satisfied)\n"
                "- time_scope: one of ['all','today','yesterday','last_week','this_month','last_30_days']\n"
                "- topic: short topic keyword (e.g. 'waiting time','staff','service','pricing') or null\n"
                "- limit: small integer (3-10) for how many example rows are useful\n\n"
                "Rules:\n"
                "- If user asks for NPS or score, prefer query_type='nps_metric'.\n"
                "- If user asks to see feedback texts, use query_type='examples'.\n"
                "- If user asks 'why', 'what is happening', or wants explanation, prefer 'insight' or 'complex_reasoning'.\n"
                "- If user mentions a specific customer and their visits/history, use query_type='customer_trace'.\n"
                "- If user compares two branches/areas, use query_type='compare' and set both branch and branch_b."
            ),
        ),
        ("user", "{question}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an analytics assistant for customer feedback.\n"
                "You will receive:\n"
                "- original_question: what the user asked\n"
                "- intent: JSON representing parsed intent\n"
                "- metrics: JSON with numeric metrics and aggregates\n"
                "- samples: short feedback examples (already filtered)\n\n"
                "Use ONLY this data. Do NOT invent new numbers.\n"
                "If metrics are empty or counts are zero, say clearly that there is not enough data.\n"
                "Answer in 30-60 words. Be clear and concise."
            ),
        ),
        (
            "user",
            "original_question:\n{question}\n\nintent:\n{intent_json}\n\nmetrics:\n{metrics_json}\n\nsamples:\n{samples_text}",
        ),
    ]
)


class FeedbackChatbotService:
    def __init__(self, model_name: str = "gpt-4.1-mini") -> None:
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self._branch_cache: Dict[Tuple[int, int], List[str]] = {}

    def answer_question(
        self,
        question: str,
        client_id: int,
        format_id: int,
    ) -> Dict[str, Any]:
        question = (question or "").strip()
        if not question:
            return {
                "answer": "Please provide a question about your feedback data.",
                "intent": {},
                "metrics": {},
                "samples": [],
            }
        intent = self._extract_intent(question)
        date_range = self._resolve_date_range(question, intent)
        query_type = intent.get("query_type") or "insight"
        sentiment = intent.get("sentiment") or "all"
        topic = intent.get("topic")
        branch_hint = intent.get("branch")
        branch_b_hint = intent.get("branch_b")
        customer_hint = intent.get("customer")
        limit_raw = intent.get("limit")
        try:
            limit = int(limit_raw) if limit_raw is not None else 5
        except Exception:
            limit = 5
        limit = max(1, min(limit, 20))
        metrics: Dict[str, Any] = {}
        samples: List[Dict[str, Any]] = []
        if query_type == "customer_trace":
            metrics = self._compute_customer_metrics(
                client_id=client_id,
                format_id=format_id,
                customer_hint=customer_hint,
                date_range=date_range,
            )
            samples = self._fetch_examples(
                client_id=client_id,
                format_id=format_id,
                branch_hint=None,
                sentiment="all",
                date_range=date_range,
                topic=None,
                customer_hint=customer_hint,
                limit=limit,
            )
        elif query_type == "compare":
            branch_a_name = self._match_branch(branch_hint, client_id, format_id)
            branch_b_name = self._match_branch(branch_b_hint, client_id, format_id)
            metrics_a = self._compute_nps_metrics(
                client_id=client_id,
                format_id=format_id,
                branch_name=branch_a_name,
                sentiment=sentiment,
                topic=topic,
                date_range=date_range,
            )
            metrics_b = self._compute_nps_metrics(
                client_id=client_id,
                format_id=format_id,
                branch_name=branch_b_name,
                sentiment=sentiment,
                topic=topic,
                date_range=date_range,
            )
            metrics = {"primary_branch": metrics_a, "secondary_branch": metrics_b}
            samples = self._fetch_examples(
                client_id=client_id,
                format_id=format_id,
                branch_hint=branch_hint,
                sentiment=sentiment,
                date_range=date_range,
                topic=topic,
                customer_hint=None,
                limit=limit,
            )
        elif query_type == "nps_metric":
            branch_name = self._match_branch(branch_hint, client_id, format_id)
            metrics = self._compute_nps_metrics(
                client_id=client_id,
                format_id=format_id,
                branch_name=branch_name,
                sentiment=sentiment,
                topic=topic,
                date_range=date_range,
            )
            samples = self._fetch_examples(
                client_id=client_id,
                format_id=format_id,
                branch_hint=branch_hint,
                sentiment=sentiment,
                date_range=date_range,
                topic=topic,
                customer_hint=None,
                limit=min(limit, 5),
            )
        elif query_type == "examples":
            metrics = {}
            samples = self._fetch_examples(
                client_id=client_id,
                format_id=format_id,
                branch_hint=branch_hint,
                sentiment=sentiment,
                date_range=date_range,
                topic=topic,
                customer_hint=customer_hint,
                limit=limit,
            )
        else:
            branch_name = self._match_branch(branch_hint, client_id, format_id)
            metrics = self._compute_nps_metrics(
                client_id=client_id,
                format_id=format_id,
                branch_name=branch_name,
                sentiment=sentiment,
                topic=topic,
                date_range=date_range,
            )
            samples = self._fetch_examples(
                client_id=client_id,
                format_id=format_id,
                branch_hint=branch_hint,
                sentiment=sentiment,
                date_range=date_range,
                topic=topic,
                customer_hint=customer_hint,
                limit=limit if query_type in ("examples", "mixed") else min(limit, 10),
            )
        answer = self._generate_answer(
            question=question,
            intent=intent,
            metrics=metrics,
            samples=samples,
        )
        return {
            "answer": answer,
            "intent": intent,
            "metrics": metrics,
            "samples": samples,
        }

    def _extract_intent(self, question: str) -> Dict[str, Any]:
        try:
            msg = INTENT_PROMPT.invoke({"question": question})
            resp = self.llm.invoke(msg.to_messages())
            raw = (resp.content or "").strip()
            raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
            raw = re.sub(r"```$", "", raw).strip()
            data = json.loads(raw)
        except Exception:
            q = question.lower()
            query_type = "insight"
            if "nps" in q or "score" in q:
                query_type = "nps_metric"
            elif "example" in q or "show" in q or "list" in q:
                query_type = "examples"
            elif "customer" in q:
                query_type = "customer_trace"
            data = {
                "query_type": query_type,
                "branch": None,
                "branch_b": None,
                "customer": None,
                "sentiment": "all",
                "time_scope": "all",
                "topic": None,
                "limit": 5,
            }
        data.setdefault("query_type", "insight")
        data.setdefault("branch", None)
        data.setdefault("branch_b", None)
        data.setdefault("customer", None)
        data.setdefault("sentiment", "all")
        data.setdefault("time_scope", "all")
        data.setdefault("topic", None)
        data.setdefault("limit", 5)
        return data

    def _resolve_date_range(
        self, question: str, intent: Dict[str, Any]
    ) -> Optional[Tuple[str, str]]:
        explicit = _parse_explicit_date(question)
        if explicit:
            return explicit
        scope = intent.get("time_scope") or "all"
        return _detect_date_range_from_scope(scope)

    def _get_branches(self, client_id: int, format_id: int) -> List[str]:
        key = (client_id, format_id)
        if key in self._branch_cache:
            return self._branch_cache[key]
        sql = text(
            """
            SELECT DISTINCT hierarchy_name
            FROM audit_table
            WHERE client_id = :client_id
              AND format_id = :format_id
              AND hierarchy_name IS NOT NULL
            """
        )
        try:
            with engine.connect() as conn:
                rows = conn.execute(sql, {"client_id": client_id, "format_id": format_id}).fetchall()
        except SQLAlchemyError:
            rows = []
        branches = [r[0] for r in rows]
        self._branch_cache[key] = branches
        return branches

    def _match_branch(
        self,
        hint: Optional[str],
        client_id: int,
        format_id: int,
    ) -> Optional[str]:
        if not hint:
            return None
        hint_norm = _normalize_branch(hint)
        if not hint_norm:
            return None
        branches = self._get_branches(client_id, format_id)
        if not branches:
            return None
        norm_map: Dict[str, str] = {}
        norm_list: List[str] = []
        for b in branches:
            n = _normalize_branch(b)
            if not n:
                continue
            norm_map[n] = b
            norm_list.append(n)
        hint_tokens = hint_norm.split()
        hint_sorted = " ".join(sorted(hint_tokens))
        candidates = {hint_norm, hint_sorted}
        best_norm = None
        for cand in candidates:
            best_matches = [n for n in norm_list if n == cand]
            if best_matches:
                best_norm = best_matches[0]
                break
        if not best_norm:
            from difflib import get_close_matches

            for cand in candidates:
                matches = get_close_matches(cand, norm_list, n=1, cutoff=0.7)
                if matches:
                    best_norm = matches[0]
                    break
        if not best_norm:
            return None
        return norm_map.get(best_norm)

    def _compute_nps_metrics(
        self,
        client_id: int,
        format_id: int,
        branch_name: Optional[str],
        sentiment: str,
        topic: Optional[str],
        date_range: Optional[Tuple[str, str]],
    ) -> Dict[str, Any]:
        filters = [
            "client_id = :client_id",
            "format_id = :format_id",
        ]
        params: Dict[str, Any] = {
            "client_id": client_id,
            "format_id": format_id,
        }
        if branch_name:
            filters.append("hierarchy_name = :branch_name")
            params["branch_name"] = branch_name
        if sentiment == "negative":
            filters.append("is_negative = 1")
        elif sentiment == "positive":
            filters.append("is_positive = 1")
        if topic:
            filters.append("topic = :topic")
            params["topic"] = topic
        if date_range:
            filters.append("date_time BETWEEN :start_dt AND :end_dt")
            params["start_dt"], params["end_dt"] = date_range
        where_clause = " AND ".join(filters)
        sql = text(
            f"""
            SELECT nps_category
            FROM audit_table
            WHERE {where_clause}
              AND nps_category IS NOT NULL
            """
        )
        try:
            with engine.connect() as conn:
                rows = conn.execute(sql, params).fetchall()
        except (SQLAlchemyError, SQLATimeout):
            return {
                "branch": branch_name,
                "total_responses": 0,
                "promoters": 0,
                "detractors": 0,
                "passives": 0,
                "nps": None,
            }
        total = len(rows)
        promoters = 0
        detractors = 0
        for r in rows:
            cat = (r[0] or "").strip().lower()
            if cat == "promoter":
                promoters += 1
            elif cat == "detractor":
                detractors += 1
        passives = total - promoters - detractors
        if total > 0:
            nps_value = (promoters / total - detractors / total) * 100.0
        else:
            nps_value = None
        return {
            "branch": branch_name,
            "total_responses": total,
            "promoters": promoters,
            "detractors": detractors,
            "passives": passives,
            "nps": round(nps_value, 2) if nps_value is not None else None,
        }

    def _compute_customer_metrics(
        self,
        client_id: int,
        format_id: int,
        customer_hint: Optional[str],
        date_range: Optional[Tuple[str, str]],
    ) -> Dict[str, Any]:
        filters = [
            "client_id = :client_id",
            "format_id = :format_id",
        ]
        params: Dict[str, Any] = {
            "client_id": client_id,
            "format_id": format_id,
        }
        if customer_hint:
            filters.append("customer_name = :customer_name")
            params["customer_name"] = customer_hint
        if date_range:
            filters.append("date_time BETWEEN :start_dt AND :end_dt")
            params["start_dt"], params["end_dt"] = date_range
        where_clause = " AND ".join(filters)
        sql = text(
            f"""
            SELECT 
                COUNT(*) AS total_visits,
                COUNT(DISTINCT hierarchy_name) AS distinct_branches,
                MIN(date_time) AS first_visit,
                MAX(date_time) AS last_visit
            FROM audit_table
            WHERE {where_clause}
            """
        )
        try:
            with engine.connect() as conn:
                row = conn.execute(sql, params).fetchone()
        except (SQLAlchemyError, SQLATimeout):
            row = None
        if not row:
            return {
                "customer": customer_hint,
                "total_visits": 0,
                "distinct_branches": 0,
                "first_visit": None,
                "last_visit": None,
            }
        return {
            "customer": customer_hint,
            "total_visits": int(row[0] or 0),
            "distinct_branches": int(row[1] or 0),
            "first_visit": str(row[2]) if row[2] is not None else None,
            "last_visit": str(row[3]) if row[3] is not None else None,
        }

    def _fetch_examples(
        self,
        client_id: int,
        format_id: int,
        branch_hint: Optional[str],
        sentiment: str,
        date_range: Optional[Tuple[str, str]],
        topic: Optional[str],
        customer_hint: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        filters = [
            "client_id = :client_id",
            "format_id = :format_id",
        ]
        params: Dict[str, Any] = {
            "client_id": client_id,
            "format_id": format_id,
            "limit": max(1, min(limit, 20)),
        }
        branch_name = self._match_branch(branch_hint, client_id, format_id)
        if branch_name:
            filters.append("hierarchy_name = :branch_name")
            params["branch_name"] = branch_name
        if sentiment == "negative":
            filters.append("is_negative = 1")
        elif sentiment == "positive":
            filters.append("is_positive = 1")
        if date_range:
            filters.append("date_time BETWEEN :start_dt AND :end_dt")
            params["start_dt"], params["end_dt"] = date_range
        if topic:
            filters.append("topic = :topic")
            params["topic"] = topic
        if customer_hint:
            filters.append("customer_name = :customer_name")
            params["customer_name"] = customer_hint
        where_clause = " AND ".join(filters)
        sql = text(
            f"""
            SELECT date_time, hierarchy_name, nps_category, topic, summary
            FROM audit_table
            WHERE {where_clause}
            ORDER BY date_time DESC
            LIMIT :limit
            """
        )
        try:
            with engine.connect() as conn:
                rows = conn.execute(sql, params).fetchall()
        except (SQLAlchemyError, SQLATimeout):
            return []
        examples: List[Dict[str, Any]] = []
        for dt, branch, nps_cat, topic_val, summary in rows:
            text_value = ""
            if summary is not None:
                s = str(summary)
                m = re.search(r"\[Text:\s*(.*)\]$", s, flags=re.DOTALL)
                if m:
                    text_value = m.group(1).strip()
                else:
                    text_value = s.strip()
            if len(text_value) > 260:
                text_value = text_value[:257] + "..."
            examples.append(
                {
                    "date_time": str(dt) if dt is not None else None,
                    "branch": branch,
                    "nps_category": nps_cat,
                    "topic": topic_val,
                    "text": text_value,
                }
            )
        return examples

    def _generate_answer(
        self,
        question: str,
        intent: Dict[str, Any],
        metrics: Dict[str, Any],
        samples: List[Dict[str, Any]],
    ) -> str:
        try:
            metrics_json = json.dumps(metrics or {}, ensure_ascii=False)
        except Exception:
            metrics_json = "{}"
        try:
            intent_json = json.dumps(intent or {}, ensure_ascii=False)
        except Exception:
            intent_json = "{}"
        lines: List[str] = []
        for s in samples[:10]:
            line = (
                f"- [{s.get('date_time','')}] "
                f"[{s.get('branch','')}] "
                f"[{s.get('nps_category','')}] "
                f"[{s.get('topic','')}] "
                f"{(s.get('text') or '')}"
            )
            if len(line) > 320:
                line = line[:317] + "..."
            lines.append(line)
        samples_text = "\n".join(lines)
        msg = ANSWER_PROMPT.invoke(
            {
                "question": question,
                "intent_json": intent_json,
                "metrics_json": metrics_json,
                "samples_text": samples_text,
            }
        )
        try:
            resp = self.llm.invoke(msg.to_messages())
            return (resp.content or "").strip()
        except Exception:
            if isinstance(metrics, dict) and metrics.get("nps") is not None:
                return f"NPS is {metrics['nps']} based on {metrics.get('total_responses',0)} responses."
            if samples:
                texts = [s.get("text", "") for s in samples[:3]]
                return "Here are some relevant feedback examples:\n" + "\n".join(texts)
            return "I could not compute an answer from the available data."


chatbot_service = FeedbackChatbotService()

# from __future__ import annotations

# import re
# import time
# from datetime import date, timedelta
# from typing import List, Tuple, Dict, Optional

# from sqlalchemy import text
# from sqlalchemy.exc import SQLAlchemyError, TimeoutError as SQLATimeout

# from config import engine
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_community.chat_message_histories import ChatMessageHistory


# ALLOWED_COLS = {
#     "id", "client_id", "format_id", "hierarchy_id", "hierarchy_name",
#     "customer_name", "customer_mobile", "feedback_id", "date_time",
#     "nps", "average_rating", "summary", "nps_category", "month",
#     "week_start", "is_positive", "is_negative", "sentiment_score",
#     "topic", "flagged", "keywords", "created", "updated"
# }
# MAX_ROWS = 10

# SELECT_SINGLE_STMT = re.compile(
#     r"^\s*select\b[\s\S]+?\bfrom\b\s+audit_table\b(?![\s\S]*\bselect\b)[\s\S]*$",
#     re.IGNORECASE,
# )
# FORBIDDEN = re.compile(
#     r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|call|into\s+outfile|load\s+data|union|with)\b",
#     re.IGNORECASE,
# )
# AGG_FUNC = re.compile(r"\b(avg|count|min|max|sum)\s*\(", re.IGNORECASE)

# CODE_FENCE = re.compile(r"^\s*```(?:sql)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)
# LINE_COMMENT = re.compile(r"--[^\n]*")
# BLOCK_COMMENT = re.compile(r"/\*[\s\S]*?\*/")

# DML_PATTERN = re.compile(
#     r"\b(update|delete|insert|truncate|alter|drop|create|replace)\b",
#     re.IGNORECASE,
# )

# GREETING_SET = {
#     "hi", "hello", "hey", "hlo", "yo",
#     "ok", "okay", "hmm", "hmmm", "nf",
#     "thanks", "thank you",
# }
# INTENT_KEYWORDS = {
#     "feedback", "rating", "ratings", "summary", "summaries",
#     "experience", "experiences", "issue", "issues",
#     "problem", "problems", "sentiment", "score", "scores",
#     "nps", "branch", "branches", "performance",
# }

# _SESSION_STORE: Dict[str, ChatMessageHistory] = {}


# def _normalize_sql(raw: str) -> str:
#     s = raw or ""
#     s = CODE_FENCE.sub("", s)
#     s = BLOCK_COMMENT.sub("", s)
#     s = LINE_COMMENT.sub("", s)
#     s = s.strip()
#     if "SQL:" in s[:10].upper():
#         s = s.split(":", 1)[-1].strip()
#     if ";" in s:
#         s = s.split(";", 1)[0]
#     return s.strip()


# def _cond_present(sql: str, col: str, param_name: str) -> bool:
#     pattern = rf"\b{col}\s*=\s*(?:[:]?" + re.escape(param_name) + r"|\d+)"
#     return re.search(pattern, sql, re.IGNORECASE) is not None


# def _append_conditions(sql: str, needed_conds: List[str]) -> str:
#     tail = re.search(r"\b(order|group|limit)\b", sql, re.IGNORECASE)
#     insert_at = tail.start() if tail else len(sql)
#     prefix = sql[:insert_at].rstrip()
#     suffix = sql[insert_at:]
#     cond_str = " AND ".join(needed_conds)
#     if re.search(r"\bwhere\b", prefix, re.IGNORECASE):
#         new_sql = f"{prefix} AND {cond_str}"
#     else:
#         sep = " " if not prefix.endswith((" ", "\n", "\t")) else ""
#         new_sql = f"{prefix}{sep}WHERE {cond_str}"
#     if suffix and not suffix.startswith((" ", "\n", "\t")):
#         suffix = " " + suffix
#     return new_sql + suffix


# def _enforce_limit(sql: str) -> str:
#     m = re.search(r"\blimit\s+(\d+)", sql, re.IGNORECASE)
#     if m:
#         if int(m.group(1)) > MAX_ROWS:
#             sql = re.sub(r"\blimit\s+\d+", f"LIMIT {MAX_ROWS}", sql, flags=re.IGNORECASE)
#     else:
#         if not sql.rstrip().endswith((" ", "\n", "\t")):
#             sql += " "
#         sql += f"LIMIT {MAX_ROWS}"
#     return sql


# def _detect_date_range(question: str) -> Optional[Tuple[str, str]]:
#     q = (question or "").lower()
#     today = date.today()
#     start: Optional[date] = None
#     end: Optional[date] = None

#     if "today" in q:
#         start = today
#         end = today + timedelta(days=1)
#     elif "yesterday" in q:
#         start = today - timedelta(days=1)
#         end = today
#     elif "last week" in q:
#         start = today - timedelta(days=7)
#         end = today + timedelta(days=1)
#     elif "last 4 months" in q or "last four months" in q:
#         start = today - timedelta(days=120)
#         end = today + timedelta(days=1)
#     elif "this month" in q:
#         start = today.replace(day=1)
#         if start.month == 12:
#             end = date(start.year + 1, 1, 1)
#         else:
#             end = date(start.year, start.month + 1, 1)
#     elif "recent" in q:
#         start = today - timedelta(days=14)
#         end = today + timedelta(days=1)

#     if not start or not end:
#         return None

#     start_str = f"{start.isoformat()} 00:00:00"
#     end_str = f"{end.isoformat()} 00:00:00"
#     return start_str, end_str


# def _force_client_and_format_filter(sql: str, client_id: int, format_id: int) -> Tuple[str, Dict[str, object]]:
#     sql = _normalize_sql(sql)

#     if FORBIDDEN.search(sql):
#         raise ValueError("Only safe SELECT queries are allowed. Database changes are not permitted.")

#     if AGG_FUNC.search(sql):
#         raise ValueError(
#             "Aggregations like AVG, COUNT, SUM, MIN, MAX are not supported. "
#             "Please ask for individual records or use stored columns such as nps or average_rating."
#         )

#     if not SELECT_SINGLE_STMT.match(sql):
#         raise ValueError("Query must be a single SELECT from audit_table without subqueries or additional SELECTs.")

#     if re.search(r"\bjoin\b", sql, re.IGNORECASE):
#         raise ValueError("JOINs are not allowed. Please ask using only audit_table.")

#     m = re.search(r"select\s+(.*?)\s+from\s+audit_table", sql, re.IGNORECASE | re.DOTALL)
#     if m:
#         cols = m.group(1).strip()
#         if cols not in ("*", "*, *"):
#             raw_cols = [c.strip() for c in cols.split(",")]
#             for c in raw_cols:
#                 base = re.split(r"\s+as\s+|\s+", c, flags=re.IGNORECASE)[0]
#                 base = base.replace("`", "").replace('"', "")
#                 if "(" in base or ")" in base:
#                     raise ValueError(
#                         "SQL functions in the SELECT list are not supported. "
#                         "Please request raw columns only."
#                     )
#                 if "." in base:
#                     _, base = base.split(".", 1)
#                 if base.lower() not in ALLOWED_COLS:
#                     raise ValueError(f"Column not allowed: {base}")

#     needed: List[str] = []
#     if not _cond_present(sql, "client_id", "client_id"):
#         needed.append("client_id = :client_id")
#     if not _cond_present(sql, "format_id", "format_id"):
#         needed.append("format_id = :format_id")

#     if needed:
#         sql = _append_conditions(sql, needed)

#     sql = _enforce_limit(sql)
#     params: Dict[str, object] = {"client_id": int(client_id), "format_id": int(format_id)}
#     return sql, params


# def _summaries_from(rows, cols: List[str]) -> List[str]:
#     cols_lower = [c.lower() for c in cols]
#     if "summary" in cols_lower:
#         sidx = cols_lower.index("summary")
#         if "hierarchy_name" in cols_lower:
#             bidx = cols_lower.index("hierarchy_name")
#             out = [f"{rows[i][bidx]} — {rows[i][sidx]}" for i in range(len(rows))]
#         else:
#             out = [str(r[sidx]) for r in rows]
#     else:
#         out = [str(r[0]) for r in rows]
#     return out[:20]


# def _extract_nps_values(rows, cols: List[str]) -> List[float]:
#     cols_lower = [c.lower() for c in cols]
#     if "nps" not in cols_lower:
#         return []
#     idx = cols_lower.index("nps")
#     values: List[float] = []
#     for r in rows:
#         try:
#             v = r[idx]
#             if v is None:
#                 continue
#             v = float(v)
#             values.append(v)
#         except Exception:
#             continue
#     return values


# def _compute_nps_score(values: List[float]) -> Optional[float]:
#     if not values:
#         return None
#     avg = sum(values) / len(values)
#     return round(avg, 1)


# def _is_dml_question(question: str) -> bool:
#     return bool(DML_PATTERN.search(question or ""))


# def _is_irrelevant_question(question: str) -> bool:
#     q = (question or "").strip().lower()
#     if not q:
#         return True
#     if q in GREETING_SET:
#         return True
#     words = re.findall(r"\w+", q)
#     has_intent = any(k in q for k in INTENT_KEYWORDS)
#     if len(words) <= 2 and not has_intent:
#         return True
#     if len(words) > 3 and not has_intent:
#         return True
#     return False


# _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# _sql_prompt = ChatPromptTemplate.from_messages([
#     MessagesPlaceholder("history"),
#     ("system",
#      "You are an SQL generator for a reporting chatbot. "
#      "Generate a single safe SQL SELECT on audit_table only. "
#      "No joins, subqueries, UNION, WITH, or any DML. "
#      "Use only columns from: {allowed_cols}. "
#      "Use hierarchy_name to represent branches or locations (for example Karachi – DHA). "
#      "Do not compare customer_name to a branch name. "
#      "If the user refers to time periods like days, weeks, or months, prefer the date_time column for filtering. "
#      "Always include WHERE client_id = {client_id} AND format_id = {format_id}. "
#      f"Always end the query with LIMIT {MAX_ROWS}. "
#      "Return only the SQL text, nothing else."),
#     ("human", "User Question: {question}\nSQL:"),
# ])
# _sql_chain = _sql_prompt | _llm | StrOutputParser()

# _answer_prompt = ChatPromptTemplate.from_messages([
#     MessagesPlaceholder("history"),
#     ("system",
#      "You are a professional analyst. Write a 30–40 word neutral, factual answer "
#      "based only on the provided rows and optional NPS value. "
#      "If computed_nps is a number (not None), begin with: "
#      "'The NPS score is <computed_nps>.' then briefly describe key themes. "
#      "Do not mention SQL, tables, or databases."),
#     ("human",
#      "User Question: {question}\n"
#      "Rows: {rows}\n"
#      "Computed NPS: {computed_nps}\n"
#      "Answer:"),
# ])
# _answer_chain = _answer_prompt | _llm | StrOutputParser()


# def _get_history(session_id: str) -> ChatMessageHistory:
#     if session_id not in _SESSION_STORE:
#         _SESSION_STORE[session_id] = ChatMessageHistory()
#     return _SESSION_STORE[session_id]


# def _history_from_cfg(cfg) -> ChatMessageHistory:
#     if isinstance(cfg, dict):
#         sid = (cfg.get("configurable") or {}).get("session_id", "default")
#     else:
#         sid = "default"
#     return _get_history(sid)


# _sql_chain_hist = RunnableWithMessageHistory(
#     _sql_chain,
#     _history_from_cfg,
#     input_messages_key="question",
#     history_messages_key="history",
# )

# _answer_chain_hist = RunnableWithMessageHistory(
#     _answer_chain,
#     _history_from_cfg,
#     input_messages_key="question",
#     history_messages_key="history",
# )


# def _build_fallback_sql(
#     client_id: int,
#     format_id: int,
#     date_range: Optional[Tuple[str, str]],
# ) -> Tuple[str, Dict[str, object]]:
#     preferred_cols = [
#         "hierarchy_name", "summary", "nps", "average_rating",
#         "nps_category", "sentiment_score", "date_time", "topic",
#     ]
#     cols = [c for c in preferred_cols if c in ALLOWED_COLS] or ["*"]
#     base = f"SELECT {', '.join(cols)} FROM audit_table"
#     conds = ["client_id = :client_id", "format_id = :format_id"]
#     params: Dict[str, object] = {"client_id": int(client_id), "format_id": int(format_id)}

#     if date_range:
#         conds.append("date_time >= :date_from")
#         conds.append("date_time < :date_to")
#         params["date_from"], params["date_to"] = date_range

#     sql = f"{base} WHERE {' AND '.join(conds)}"
#     sql = _enforce_limit(sql)
#     return sql, params


# def process_question_langchain(
#     question: str,
#     client_id: int,
#     format_id: int,
#     session_id: Optional[str] = None,
# ) -> dict:
#     if client_id is None:
#         raise ValueError("client_id is required")
#     if format_id is None:
#         raise ValueError("format_id is required")

#     t0 = time.time()
#     cfg = {"configurable": {"session_id": session_id or "default"}}

#     if _is_dml_question(question):
#         latency_ms = int((time.time() - t0) * 1000)
#         return {
#             "sql": None,
#             "answer": "Database changes are not allowed. Only information retrieval queries are supported.",
#             "rows": [],
#             "columns": [],
#             "computed_nps": None,
#             "latency_ms": latency_ms,
#         }

#     if _is_irrelevant_question(question):
#         latency_ms = int((time.time() - t0) * 1000)
#         return {
#             "sql": None,
#             "answer": "Please ask a clear question related to feedback, branches, ratings, sentiment, or NPS.",
#             "rows": [],
#             "columns": [],
#             "computed_nps": None,
#             "latency_ms": latency_ms,
#         }

#     date_range = _detect_date_range(question)

#     try:
#         raw_sql = _sql_chain_hist.invoke(
#             {
#                 "question": question,
#                 "allowed_cols": ", ".join(sorted(ALLOWED_COLS)),
#                 "client_id": client_id,
#                 "format_id": format_id,
#             },
#             config=cfg,
#         )
#     except Exception:
#         safe_sql, params = _build_fallback_sql(client_id, format_id, date_range)
#     else:
#         try:
#             safe_sql, params = _force_client_and_format_filter(raw_sql, client_id, format_id)
#         except ValueError as e:
#             msg = str(e)
#             if "Aggregations like AVG" in msg:
#                 latency_ms = int((time.time() - t0) * 1000)
#                 return {
#                     "sql": _normalize_sql(raw_sql),
#                     "answer": msg,
#                     "rows": [],
#                     "columns": [],
#                     "computed_nps": None,
#                     "latency_ms": latency_ms,
#                 }
#             safe_sql, params = _build_fallback_sql(client_id, format_id, date_range)

#     if date_range and "date_time" not in safe_sql.lower():
#         safe_sql = _append_conditions(safe_sql, ["date_time >= :date_from", "date_time < :date_to"])
#         params["date_from"], params["date_to"] = date_range

#     safe_sql = _enforce_limit(safe_sql)

#     try:
#         with engine.connect() as conn:
#             result = conn.execute(text(safe_sql), params)
#             rows = result.fetchall()
#             cols = list(result.keys())
#     except (SQLATimeout, SQLAlchemyError):
#         latency_ms = int((time.time() - t0) * 1000)
#         return {
#             "sql": safe_sql,
#             "answer": "A database error occurred while processing your request.",
#             "rows": [],
#             "columns": [],
#             "computed_nps": None,
#             "latency_ms": latency_ms,
#         }

#     if not rows:
#         latency_ms = int((time.time() - t0) * 1000)
#         return {
#             "sql": safe_sql,
#             "answer": "No relevant results were found for your request.",
#             "rows": [],
#             "columns": cols,
#             "computed_nps": None,
#             "latency_ms": latency_ms,
#         }

#     summaries = _summaries_from(rows, cols)
#     q_lower = (question or "").lower()
#     nps_values = _extract_nps_values(rows, cols)
#     computed_nps: Optional[float] = None
#     if "nps" in q_lower and nps_values:
#         computed_nps = _compute_nps_score(nps_values)

#     answer = _answer_chain_hist.invoke(
#         {
#             "question": question,
#             "rows": summaries,
#             "computed_nps": computed_nps,
#         },
#         config=cfg,
#     ).strip()

#     row_dicts = [dict(zip(cols, r)) for r in rows]
#     latency_ms = int((time.time() - t0) * 1000)
#     return {
#         "sql": safe_sql,
#         "answer": answer,
#         "rows": row_dicts,
#         "columns": cols,
#         "computed_nps": computed_nps,
#         "latency_ms": latency_ms,
#     }


# def process_question(
#     question: str,
#     client_id: int,
#     format_id: int,
#     session_id: Optional[str] = None,
# ) -> dict:
#     return process_question_langchain(question, client_id, format_id, session_id)


# __all__ = ["process_question", "process_question_langchain"]
