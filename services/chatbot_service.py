from __future__ import annotations

import re
import time
from datetime import date, timedelta
from typing import List, Tuple, Dict, Optional

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError, TimeoutError as SQLATimeout

from config import engine
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


ALLOWED_COLS = {
    "id", "client_id", "format_id", "hierarchy_id", "hierarchy_name",
    "customer_name", "customer_mobile", "feedback_id", "date_time",
    "nps", "average_rating", "summary", "nps_category", "month",
    "week_start", "is_positive", "is_negative", "sentiment_score",
    "topic", "flagged", "keywords", "created", "updated"
}
MAX_ROWS = 10

SELECT_SINGLE_STMT = re.compile(
    r"^\s*select\b[\s\S]+?\bfrom\b\s+audit_table\b(?![\s\S]*\bselect\b)[\s\S]*$",
    re.IGNORECASE,
)
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|call|into\s+outfile|load\s+data|union|with)\b",
    re.IGNORECASE,
)
AGG_FUNC = re.compile(r"\b(avg|count|min|max|sum)\s*\(", re.IGNORECASE)

CODE_FENCE = re.compile(r"^\s*```(?:sql)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)
LINE_COMMENT = re.compile(r"--[^\n]*")
BLOCK_COMMENT = re.compile(r"/\*[\s\S]*?\*/")

DML_PATTERN = re.compile(
    r"\b(update|delete|insert|truncate|alter|drop|create|replace)\b",
    re.IGNORECASE,
)

GREETING_SET = {
    "hi", "hello", "hey", "hlo", "yo",
    "ok", "okay", "hmm", "hmmm", "nf",
    "thanks", "thank you",
}
INTENT_KEYWORDS = {
    "feedback", "rating", "ratings", "summary", "summaries",
    "experience", "experiences", "issue", "issues",
    "problem", "problems", "sentiment", "score", "scores",
    "nps", "branch", "branches", "performance",
}

_SESSION_STORE: Dict[str, ChatMessageHistory] = {}


def _normalize_sql(raw: str) -> str:
    s = raw or ""
    s = CODE_FENCE.sub("", s)
    s = BLOCK_COMMENT.sub("", s)
    s = LINE_COMMENT.sub("", s)
    s = s.strip()
    if "SQL:" in s[:10].upper():
        s = s.split(":", 1)[-1].strip()
    if ";" in s:
        s = s.split(";", 1)[0]
    return s.strip()


def _cond_present(sql: str, col: str, param_name: str) -> bool:
    pattern = rf"\b{col}\s*=\s*(?:[:]?" + re.escape(param_name) + r"|\d+)"
    return re.search(pattern, sql, re.IGNORECASE) is not None


def _append_conditions(sql: str, needed_conds: List[str]) -> str:
    tail = re.search(r"\b(order|group|limit)\b", sql, re.IGNORECASE)
    insert_at = tail.start() if tail else len(sql)
    prefix = sql[:insert_at].rstrip()
    suffix = sql[insert_at:]
    cond_str = " AND ".join(needed_conds)
    if re.search(r"\bwhere\b", prefix, re.IGNORECASE):
        new_sql = f"{prefix} AND {cond_str}"
    else:
        sep = " " if not prefix.endswith((" ", "\n", "\t")) else ""
        new_sql = f"{prefix}{sep}WHERE {cond_str}"
    if suffix and not suffix.startswith((" ", "\n", "\t")):
        suffix = " " + suffix
    return new_sql + suffix


def _enforce_limit(sql: str) -> str:
    m = re.search(r"\blimit\s+(\d+)", sql, re.IGNORECASE)
    if m:
        if int(m.group(1)) > MAX_ROWS:
            sql = re.sub(r"\blimit\s+\d+", f"LIMIT {MAX_ROWS}", sql, flags=re.IGNORECASE)
    else:
        if not sql.rstrip().endswith((" ", "\n", "\t")):
            sql += " "
        sql += f"LIMIT {MAX_ROWS}"
    return sql


def _detect_date_range(question: str) -> Optional[Tuple[str, str]]:
    q = (question or "").lower()
    today = date.today()
    start: Optional[date] = None
    end: Optional[date] = None

    if "today" in q:
        start = today
        end = today + timedelta(days=1)
    elif "yesterday" in q:
        start = today - timedelta(days=1)
        end = today
    elif "last week" in q:
        start = today - timedelta(days=7)
        end = today + timedelta(days=1)
    elif "last 4 months" in q or "last four months" in q:
        start = today - timedelta(days=120)
        end = today + timedelta(days=1)
    elif "this month" in q:
        start = today.replace(day=1)
        if start.month == 12:
            end = date(start.year + 1, 1, 1)
        else:
            end = date(start.year, start.month + 1, 1)
    elif "recent" in q:
        start = today - timedelta(days=14)
        end = today + timedelta(days=1)

    if not start or not end:
        return None

    start_str = f"{start.isoformat()} 00:00:00"
    end_str = f"{end.isoformat()} 00:00:00"
    return start_str, end_str


def _force_client_and_format_filter(sql: str, client_id: int, format_id: int) -> Tuple[str, Dict[str, object]]:
    sql = _normalize_sql(sql)

    if FORBIDDEN.search(sql):
        raise ValueError("Only safe SELECT queries are allowed. Database changes are not permitted.")

    if AGG_FUNC.search(sql):
        raise ValueError(
            "Aggregations like AVG, COUNT, SUM, MIN, MAX are not supported. "
            "Please ask for individual records or use stored columns such as nps or average_rating."
        )

    if not SELECT_SINGLE_STMT.match(sql):
        raise ValueError("Query must be a single SELECT from audit_table without subqueries or additional SELECTs.")

    if re.search(r"\bjoin\b", sql, re.IGNORECASE):
        raise ValueError("JOINs are not allowed. Please ask using only audit_table.")

    m = re.search(r"select\s+(.*?)\s+from\s+audit_table", sql, re.IGNORECASE | re.DOTALL)
    if m:
        cols = m.group(1).strip()
        if cols not in ("*", "*, *"):
            raw_cols = [c.strip() for c in cols.split(",")]
            for c in raw_cols:
                base = re.split(r"\s+as\s+|\s+", c, flags=re.IGNORECASE)[0]
                base = base.replace("`", "").replace('"', "")
                if "(" in base or ")" in base:
                    raise ValueError(
                        "SQL functions in the SELECT list are not supported. "
                        "Please request raw columns only."
                    )
                if "." in base:
                    _, base = base.split(".", 1)
                if base.lower() not in ALLOWED_COLS:
                    raise ValueError(f"Column not allowed: {base}")

    needed: List[str] = []
    if not _cond_present(sql, "client_id", "client_id"):
        needed.append("client_id = :client_id")
    if not _cond_present(sql, "format_id", "format_id"):
        needed.append("format_id = :format_id")

    if needed:
        sql = _append_conditions(sql, needed)

    sql = _enforce_limit(sql)
    params: Dict[str, object] = {"client_id": int(client_id), "format_id": int(format_id)}
    return sql, params


def _summaries_from(rows, cols: List[str]) -> List[str]:
    cols_lower = [c.lower() for c in cols]
    if "summary" in cols_lower:
        sidx = cols_lower.index("summary")
        if "hierarchy_name" in cols_lower:
            bidx = cols_lower.index("hierarchy_name")
            out = [f"{rows[i][bidx]} — {rows[i][sidx]}" for i in range(len(rows))]
        else:
            out = [str(r[sidx]) for r in rows]
    else:
        out = [str(r[0]) for r in rows]
    return out[:20]


def _extract_nps_values(rows, cols: List[str]) -> List[float]:
    cols_lower = [c.lower() for c in cols]
    if "nps" not in cols_lower:
        return []
    idx = cols_lower.index("nps")
    values: List[float] = []
    for r in rows:
        try:
            v = r[idx]
            if v is None:
                continue
            v = float(v)
            values.append(v)
        except Exception:
            continue
    return values


def _compute_nps_score(values: List[float]) -> Optional[float]:
    if not values:
        return None
    avg = sum(values) / len(values)
    return round(avg, 1)


def _is_dml_question(question: str) -> bool:
    return bool(DML_PATTERN.search(question or ""))


def _is_irrelevant_question(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return True
    if q in GREETING_SET:
        return True
    words = re.findall(r"\w+", q)
    has_intent = any(k in q for k in INTENT_KEYWORDS)
    if len(words) <= 2 and not has_intent:
        return True
    if len(words) > 3 and not has_intent:
        return True
    return False


_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_sql_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder("history"),
    ("system",
     "You are an SQL generator for a reporting chatbot. "
     "Generate a single safe SQL SELECT on audit_table only. "
     "No joins, subqueries, UNION, WITH, or any DML. "
     "Use only columns from: {allowed_cols}. "
     "Use hierarchy_name to represent branches or locations (for example Karachi – DHA). "
     "Do not compare customer_name to a branch name. "
     "If the user refers to time periods like days, weeks, or months, prefer the date_time column for filtering. "
     "Always include WHERE client_id = {client_id} AND format_id = {format_id}. "
     f"Always end the query with LIMIT {MAX_ROWS}. "
     "Return only the SQL text, nothing else."),
    ("human", "User Question: {question}\nSQL:"),
])
_sql_chain = _sql_prompt | _llm | StrOutputParser()

_answer_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder("history"),
    ("system",
     "You are a professional analyst. Write a 30–40 word neutral, factual answer "
     "based only on the provided rows and optional NPS value. "
     "If computed_nps is a number (not None), begin with: "
     "'The NPS score is <computed_nps>.' then briefly describe key themes. "
     "Do not mention SQL, tables, or databases."),
    ("human",
     "User Question: {question}\n"
     "Rows: {rows}\n"
     "Computed NPS: {computed_nps}\n"
     "Answer:"),
])
_answer_chain = _answer_prompt | _llm | StrOutputParser()


def _get_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = ChatMessageHistory()
    return _SESSION_STORE[session_id]


def _history_from_cfg(cfg) -> ChatMessageHistory:
    if isinstance(cfg, dict):
        sid = (cfg.get("configurable") or {}).get("session_id", "default")
    else:
        sid = "default"
    return _get_history(sid)


_sql_chain_hist = RunnableWithMessageHistory(
    _sql_chain,
    _history_from_cfg,
    input_messages_key="question",
    history_messages_key="history",
)

_answer_chain_hist = RunnableWithMessageHistory(
    _answer_chain,
    _history_from_cfg,
    input_messages_key="question",
    history_messages_key="history",
)


def _build_fallback_sql(
    client_id: int,
    format_id: int,
    date_range: Optional[Tuple[str, str]],
) -> Tuple[str, Dict[str, object]]:
    preferred_cols = [
        "hierarchy_name", "summary", "nps", "average_rating",
        "nps_category", "sentiment_score", "date_time", "topic",
    ]
    cols = [c for c in preferred_cols if c in ALLOWED_COLS] or ["*"]
    base = f"SELECT {', '.join(cols)} FROM audit_table"
    conds = ["client_id = :client_id", "format_id = :format_id"]
    params: Dict[str, object] = {"client_id": int(client_id), "format_id": int(format_id)}

    if date_range:
        conds.append("date_time >= :date_from")
        conds.append("date_time < :date_to")
        params["date_from"], params["date_to"] = date_range

    sql = f"{base} WHERE {' AND '.join(conds)}"
    sql = _enforce_limit(sql)
    return sql, params


def process_question_langchain(
    question: str,
    client_id: int,
    format_id: int,
    session_id: Optional[str] = None,
) -> dict:
    if client_id is None:
        raise ValueError("client_id is required")
    if format_id is None:
        raise ValueError("format_id is required")

    t0 = time.time()
    cfg = {"configurable": {"session_id": session_id or "default"}}

    if _is_dml_question(question):
        latency_ms = int((time.time() - t0) * 1000)
        return {
            "sql": None,
            "answer": "Database changes are not allowed. Only information retrieval queries are supported.",
            "rows": [],
            "columns": [],
            "computed_nps": None,
            "latency_ms": latency_ms,
        }

    if _is_irrelevant_question(question):
        latency_ms = int((time.time() - t0) * 1000)
        return {
            "sql": None,
            "answer": "Please ask a clear question related to feedback, branches, ratings, sentiment, or NPS.",
            "rows": [],
            "columns": [],
            "computed_nps": None,
            "latency_ms": latency_ms,
        }

    date_range = _detect_date_range(question)

    try:
        raw_sql = _sql_chain_hist.invoke(
            {
                "question": question,
                "allowed_cols": ", ".join(sorted(ALLOWED_COLS)),
                "client_id": client_id,
                "format_id": format_id,
            },
            config=cfg,
        )
    except Exception:
        safe_sql, params = _build_fallback_sql(client_id, format_id, date_range)
    else:
        try:
            safe_sql, params = _force_client_and_format_filter(raw_sql, client_id, format_id)
        except ValueError as e:
            msg = str(e)
            if "Aggregations like AVG" in msg:
                latency_ms = int((time.time() - t0) * 1000)
                return {
                    "sql": _normalize_sql(raw_sql),
                    "answer": msg,
                    "rows": [],
                    "columns": [],
                    "computed_nps": None,
                    "latency_ms": latency_ms,
                }
            safe_sql, params = _build_fallback_sql(client_id, format_id, date_range)

    if date_range and "date_time" not in safe_sql.lower():
        safe_sql = _append_conditions(safe_sql, ["date_time >= :date_from", "date_time < :date_to"])
        params["date_from"], params["date_to"] = date_range

    safe_sql = _enforce_limit(safe_sql)

    try:
        with engine.connect() as conn:
            result = conn.execute(text(safe_sql), params)
            rows = result.fetchall()
            cols = list(result.keys())
    except (SQLATimeout, SQLAlchemyError):
        latency_ms = int((time.time() - t0) * 1000)
        return {
            "sql": safe_sql,
            "answer": "A database error occurred while processing your request.",
            "rows": [],
            "columns": [],
            "computed_nps": None,
            "latency_ms": latency_ms,
        }

    if not rows:
        latency_ms = int((time.time() - t0) * 1000)
        return {
            "sql": safe_sql,
            "answer": "No relevant results were found for your request.",
            "rows": [],
            "columns": cols,
            "computed_nps": None,
            "latency_ms": latency_ms,
        }

    summaries = _summaries_from(rows, cols)
    q_lower = (question or "").lower()
    nps_values = _extract_nps_values(rows, cols)
    computed_nps: Optional[float] = None
    if "nps" in q_lower and nps_values:
        computed_nps = _compute_nps_score(nps_values)

    answer = _answer_chain_hist.invoke(
        {
            "question": question,
            "rows": summaries,
            "computed_nps": computed_nps,
        },
        config=cfg,
    ).strip()

    row_dicts = [dict(zip(cols, r)) for r in rows]
    latency_ms = int((time.time() - t0) * 1000)
    return {
        "sql": safe_sql,
        "answer": answer,
        "rows": row_dicts,
        "columns": cols,
        "computed_nps": computed_nps,
        "latency_ms": latency_ms,
    }


def process_question(
    question: str,
    client_id: int,
    format_id: int,
    session_id: Optional[str] = None,
) -> dict:
    return process_question_langchain(question, client_id, format_id, session_id)


__all__ = ["process_question", "process_question_langchain"]
