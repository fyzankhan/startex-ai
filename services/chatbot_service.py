from __future__ import annotations
import re
import time
from typing import List, Tuple, Dict
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError, TimeoutError as SQLATimeout
from config import engine
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# ---------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------
ALLOWED_COLS = {
    "id","client_id","format_id","hierarchy_id","hierarchy_name",
    "customer_name","customer_mobile","feedback_id","date_time",
    "nps","average_rating","summary","nps_category","month",
    "week_start","is_positive","is_negative","sentiment_score",
    "topic","flagged","keywords","created","updated"
}
MAX_ROWS = 10

SELECT_SINGLE_STMT = re.compile(
    r"^\s*select\b[\s\S]+?\bfrom\b\s+audit_table\b(?![\s\S]*\bselect\b)[\s\S]*$",
    re.IGNORECASE
)
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|call|into\s+outfile|load\s+data|union|with)\b",
    re.IGNORECASE
)
CODE_FENCE = re.compile(r"^\s*```(?:sql)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)
LINE_COMMENT = re.compile(r"--[^\n]*")
BLOCK_COMMENT = re.compile(r"/\*[\s\S]*?\*/")

_SESSION_STORE: Dict[str, ChatMessageHistory] = {}

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
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
    return re.search(rf"\b{col}\s*=\s*(?:[:]?{param_name}|\d+)", sql, re.IGNORECASE) is not None

def _append_conditions(sql: str, needed_conds: list[str]) -> str:
    tail = re.search(r"\b(order|group|limit)\b", sql, re.IGNORECASE)
    insert_at = tail.start() if tail else len(sql)
    prefix = sql[:insert_at].rstrip()
    suffix = sql[insert_at:]
    if re.search(r"\bwhere\b", prefix, re.IGNORECASE):
        new_sql = f"{prefix} AND {' AND '.join(needed_conds)}"
    else:
        sep = " " if not prefix.endswith((" ", "\n", "\t")) else ""
        new_sql = f"{prefix}{sep}WHERE {' AND '.join(needed_conds)}"
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

def _force_client_and_format_filter(sql: str, client_id: int, format_id: int) -> Tuple[str, dict]:
    sql = _normalize_sql(sql)
    if FORBIDDEN.search(sql):
        raise ValueError("Only a single safe SELECT is allowed.")
    if not SELECT_SINGLE_STMT.match(sql):
        raise ValueError("Query must be a single SELECT from audit_table without subqueries.")
    if re.search(r"\bjoin\b", sql, re.IGNORECASE):
        raise ValueError("JOINs are not allowed.")
    m = re.search(r"select\s+(.*?)\s+from\s+audit_table", sql, re.IGNORECASE | re.DOTALL)
    if m:
        cols = m.group(1).strip()
        if cols not in ("*", "*, *"):
            raw_cols = [c.strip() for c in cols.split(",")]
            for c in raw_cols:
                base = re.split(r"\s+as\s+|\s+", c, flags=re.IGNORECASE)[0]
                base = base.replace("`", "").replace('"', "")
                if "(" in base or ")" in base:
                    raise ValueError("Functions in SELECT are not allowed.")
                if "." in base:
                    _, base = base.split(".", 1)
                if base.lower() not in ALLOWED_COLS:
                    raise ValueError(f"Column not allowed: {base}")
    needed = []
    if not _cond_present(sql, "client_id", "client_id"):
        needed.append("client_id = :client_id")
    if not _cond_present(sql, "format_id", "format_id"):
        needed.append("format_id = :format_id")
    if needed:
        sql = _append_conditions(sql, needed)
    sql = _enforce_limit(sql)
    return sql, {"client_id": int(client_id), "format_id": int(format_id)}

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

# ---------------------------------------------------------------------
# LLM SETUP
# ---------------------------------------------------------------------
_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_sql_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder("history"),
    ("system",
     "Generate a single safe SQL SELECT on audit_table only. "
     "No joins, subqueries, UNION, or DML. "
     "Use only columns from: {allowed_cols}. "
     "Always include WHERE client_id = {client_id} AND format_id = {format_id}. "
     f"End with LIMIT {MAX_ROWS}. Return only SQL."),
    ("human", "User Question: {question}\nSQL:")
])
_sql_chain = _sql_prompt | _llm | StrOutputParser()

_answer_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder("history"),
    ("system",
     "You are a professional analyst. Write a 40–70 word neutral, factual summary "
     "based on the retrieved data. Do not mention SQL or internal details."),
    ("human", "User Question: {question}\nRows: {rows}\nAnswer:")
])
_answer_chain = _answer_prompt | _llm | StrOutputParser()

# ---------------------------------------------------------------------
# HISTORY MANAGEMENT (FIXED)
# ---------------------------------------------------------------------
def _get_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = ChatMessageHistory()
    return _SESSION_STORE[session_id]

_sql_chain_hist = RunnableWithMessageHistory(
    _sql_chain,
    lambda cfg: _get_history(
        (cfg.get("configurable") or {}).get("session_id", "default")
        if isinstance(cfg, dict) else "default"
    ),
    input_messages_key="question",
    history_messages_key="history",
)

_answer_chain_hist = RunnableWithMessageHistory(
    _answer_chain,
    lambda cfg: _get_history(
        (cfg.get("configurable") or {}).get("session_id", "default")
        if isinstance(cfg, dict) else "default"
    ),
    input_messages_key="question",
    history_messages_key="history",
)

# ---------------------------------------------------------------------
# MAIN PROCESSING LOGIC
# ---------------------------------------------------------------------
def process_question_langchain(
    question: str, client_id: int, format_id: int, session_id: str | None = None
) -> dict:
    if client_id is None:
        raise ValueError("client_id is required")
    if format_id is None:
        raise ValueError("format_id is required")

    cfg = {"configurable": {"session_id": session_id or "default"}}
    t0 = time.time()

    raw_sql = _sql_chain_hist.invoke(
        {"question": question,
         "allowed_cols": ", ".join(sorted(ALLOWED_COLS)),
         "client_id": client_id,
         "format_id": format_id},
        config=cfg,
    )

    safe_sql, params = _force_client_and_format_filter(raw_sql, client_id, format_id)
    try:
        with engine.connect() as conn:
            result = conn.execute(text(safe_sql), params)
            rows = result.fetchall()
            cols = list(result.keys())
    except (SQLATimeout, SQLAlchemyError) as e:
        raise RuntimeError(f"Database error: {str(e)}")

    if not rows:
        return {"sql": safe_sql, "answer": "No answer found",
                "latency_ms": int((time.time() - t0) * 1000)}

    summaries = _summaries_from(rows, cols)
    answer = _answer_chain_hist.invoke({"question": question, "rows": summaries}, config=cfg).strip()

    latency_ms = int((time.time() - t0) * 1000)
    return {"sql": safe_sql, "answer": answer, "latency_ms": latency_ms}

def process_question(question: str, client_id: int, format_id: int, session_id: str | None = None) -> dict:
    return process_question_langchain(question, client_id, format_id, session_id)

__all__ = ["process_question", "process_question_langchain"]
