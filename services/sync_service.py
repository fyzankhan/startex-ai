from datetime import datetime
from typing import List, Dict, Any
from sqlalchemy import text, bindparam
from config import engine
import pinecone
from openai import OpenAI


class SyncService:

    @staticmethod
    def sync(batch_size: int = 150):
        client = OpenAI()
        pc = pinecone.Pinecone()
        index = pc.Index("audit-nps")

        sql = """
        SELECT 
            id, client_id, format_id, customer_name,
            branch_name, city, visit_date, nps_score, summary
        FROM audit_records
        WHERE is_indexed = 0
        ORDER BY id ASC
        LIMIT :limit
        """

        with engine.connect() as conn:
            rows = conn.execute(text(sql), {"limit": batch_size}).mappings().all()

        if not rows:
            return {"synced": 0, "message": "No new rows to sync"}

        namespaces: Dict[str, List[Dict[str, Any]]] = {}

        for r in rows:
            ns = f"client_{r['client_id']}"

            text_doc = (
                f"Branch: {r['branch_name']}. "
                f"City: {r['city']}. "
                f"Customer: {r['customer_name']}. "
                f"Visit date: {r['visit_date']}. "
                f"NPS: {r['nps_score']}. "
                f"Feedback: {r['summary']}."
            )

            emb = client.embeddings.create(
                model="text-embedding-3-small",
                input=text_doc
            ).data[0].embedding

            if ns not in namespaces:
                namespaces[ns] = []

            namespaces[ns].append({
                "id": str(r["id"]),
                "values": emb,
                "metadata": {
                    "id": r["id"],
                    "client_id": r["client_id"],
                    "format_id": r["format_id"],
                    "branch_name": r["branch_name"],
                    "city": r["city"],
                    "customer_name": r["customer_name"],
                    "visit_date": str(r["visit_date"]),
                    "nps_score": r["nps_score"],
                    "summary": r["summary"]
                }
            })

        for ns, vecs in namespaces.items():
            index.upsert(vectors=vecs, namespace=ns)

        ids = [r["id"] for r in rows]

        with engine.begin() as conn:
            conn.execute(
                text("""
                    UPDATE audit_records
                    SET is_indexed = 1, indexed_at = :now
                    WHERE id IN :ids
                """).bindparams(
                    bindparam("ids", expanding=True)
                ),
                {"now": datetime.now(), "ids": ids}
            )

        return {"synced": len(rows), "message": "Sync completed"}
