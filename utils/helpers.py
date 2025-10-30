import re

def clean_sql(sql: str) -> str:
    """Remove markdown fences or unsafe content from SQL."""
    sql = sql.replace("```sql", "").replace("```", "").strip()
    sql = re.sub(r"^SQL\s*:\s*", "", sql, flags=re.IGNORECASE)
    return sql
