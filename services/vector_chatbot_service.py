from __future__ import annotations
import os
import re
import json
from typing import Any, Dict, List
from dotenv import load_dotenv

load_dotenv()

import pinecone
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """
Return STRICT JSON:
{
 "query_type": "",
 "branch": "",
 "branch_b": "",
 "customer": "",
 "topic": "",
 "sentiment": "",
 "limit": 5
}
"""),
    ("user", "{question}")
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """
Use only the provided context. Never invent numbers.
Always answer in 3–5 lines.
"""),
    ("user", "question:\n{question}\n\nintent:\n{intent_json}\n\ncontext:\n{context}")
])


class PineconeRetriever:
    def __init__(self):
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index("audit-nps")

    def retrieve(self, query: str, client_id: int, format_id: int, top_k: int = 5):
        emb = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=query
        ).data[0].embedding

        namespace = f"client_{client_id}"

        result = self.index.query(
            vector=emb,
            namespace=namespace,
            top_k=top_k * 3,
            include_metadata=True
        )

        filtered = []
        for m in result.matches:
            meta = m.metadata
            if meta.get("format_id") != format_id:
                continue
            filtered.append({
                "id": m.id,
                "score": m.score,
                "branch": meta.get("branch_name"),
                "city": meta.get("city"),
                "summary": meta.get("summary"),
                "nps_score": meta.get("nps_score"),
                "customer": meta.get("customer_name"),
                "visit_date": meta.get("visit_date")
            })
            if len(filtered) == top_k:
                break

        return filtered


class PureVectorChatbotService:
    IRRELEVANT_SINGLE_WORDS = {
        "hi","hello","hey","hlo","yo","ok","okay","k","kk","hmm","hmmm","hm",
        "yes","no","sure","thanks","thank you","thx","tnx","salam","salaam"
    }

    EMOJI_PATTERN = re.compile(r"^[\W_]+$")
    NOISE_PATTERN = re.compile(r"^[?.!/\\|,\-*+_=(){}\[\]<> ]+$")

    SMALL_TALK = [
        "how are you","what's up","whats up","kya hal","kase ho",
        "how you doing","how r u","good morning","good night","gm","gn"
    ]

    DOMAIN = [
        "nps","score","branch","city","customer","visit","feedback","summary",
        "complaint","rating","reviews","service","detractor","promoter",
        "passive","location","format","section"
    ]

    def __init__(self, model_name="gpt-4.1-mini"):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.retriever = PineconeRetriever()

    def answer(self, question: str, client_id: int, format_id: int) -> Dict[str, Any]:
        q = (question or "").strip()
        if not q:
            return {"answer": "Ask something related to feedback analytics.", "context_used": []}

        if self._irrelevant(q):
            return {
                "answer": "Hi! Please ask something about NPS, feedback, branches, customers, or insights.",
                "intent": {},
                "context_used": []
            }

        intent = self._intent(q)
        top_k = min(max(int(intent.get("limit", 5)), 1), 15)

        ctx_items = self.retriever.retrieve(q, client_id, format_id, top_k)
        ctx_text = self._context_text(ctx_items)
        answer = self._final_answer(q, intent, ctx_text)

        return {
            "answer": answer,
            "intent": intent,
            "context_used": ctx_items
        }

    def _intent(self, q: str) -> Dict[str, Any]:
        try:
            msg = INTENT_PROMPT.invoke({"question": q})
            resp = self.llm.invoke(msg.to_messages())
            raw = resp.content.strip()
            raw = re.sub(r"^```json", "", raw)
            raw = re.sub(r"```$", "", raw)
            return json.loads(raw)
        except:
            return {
                "query_type": "insight",
                "branch": None,
                "branch_b": None,
                "customer": None,
                "topic": None,
                "sentiment": "all",
                "limit": 5
            }

    def _irrelevant(self, t: str) -> bool:
        t = t.lower().strip()
        if len(t) <= 2:
            return True
        if self.EMOJI_PATTERN.fullmatch(t):
            return True
        if self.NOISE_PATTERN.fullmatch(t):
            return True
        if t in self.IRRELEVANT_SINGLE_WORDS:
            return True
        if any(p in t for p in self.SMALL_TALK):
            return True
        if not any(k in t for k in self.DOMAIN):
            if len(t.split()) <= 4:
                return True
        STOP = {
            "i","me","you","we","he","she","it","they","them","is","am","are",
            "the","a","an","this","that","to","for","in","on","at","from",
            "mera","meri","mere","tum","kya","kyu","kyun","hai","hain","ho",
            "tha","thi","hun","hoon"
        }
        words = t.split()
        if all(w in STOP for w in words):
            return True
        if any(w in t for w in ["bro","bhai","dude","buddy","yaar"]):
            return True
        return False

    def _context_text(self, items: List[Dict[str, Any]]):
        if not items:
            return "No relevant vector results found."
        out = []
        for i in items:
            out.append(
                f"- Branch: {i['branch']} | NPS: {i['nps_score']} | Customer: {i['customer']} | Summary: {i['summary']}"
            )
        return "\n".join(out)

    def _final_answer(self, question: str, intent: dict, context: str):
        msg = ANSWER_PROMPT.invoke({
            "question": question,
            "intent_json": json.dumps(intent, ensure_ascii=False),
            "context": context
        })
        try:
            resp = self.llm.invoke(msg.to_messages())
            return resp.content.strip()
        except:
            return "Unable to generate answer."


chatbot_service = PureVectorChatbotService()
