import os
import json
import pandas as pd
from sqlalchemy import text
from config import engine
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstores")
os.makedirs(VECTORSTORE_DIR, exist_ok=True)
EMBED_MODEL = "text-embedding-3-small"


def get_vector_path(client_id: int, format_id: int) -> str:
    return os.path.join(VECTORSTORE_DIR, f"client_{client_id}_format_{format_id}")


def get_meta_path(path: str) -> str:
    return path + "_meta.json"


def get_last_created_timestamp(path: str) -> str | None:
    meta_path = get_meta_path(path)
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            data = json.load(f)
            return data.get("last_created")
    return None


def update_meta(path: str, last_created: str):
    meta_path = get_meta_path(path)
    with open(meta_path, "w") as f:
        json.dump({"last_created": last_created}, f, indent=2)


def load_audit_data(client_id: int, format_id: int, last_created: str | None = None) -> pd.DataFrame:
    sql = """
        SELECT *
        FROM audit_table
        WHERE client_id = :client_id AND format_id = :format_id
    """
    if last_created:
        sql += " AND created > :last_created"
    sql += " ORDER BY created ASC"
    params = {"client_id": client_id, "format_id": format_id}
    if last_created:
        params["last_created"] = last_created
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)
    if df.empty:
        return df
    last_created_value = str(df["created"].max())
    df = df.fillna("").astype(str)

    def make_text(r):
        return (
            f"Feedback record ID {r['id']} for hierarchy '{r['hierarchy_name']}' "
            f"(hierarchy_id: {r['hierarchy_id']}) was submitted by customer {r['customer_name']} "
            f"(mobile: {r['customer_mobile']}) on {r['date_time']}. "
            f"The NPS score was {r['nps']} ({r['nps_category']}). "
            f"Sentiment score: {r['sentiment_score']}, Topic: {r['topic']}, Flagged: {r['flagged']}. "
            f"Keywords: {r['keywords']}. "
            f"Summary: {r['summary']}"
        )

    df["text"] = df.apply(make_text, axis=1)
    df["_last_created_max"] = last_created_value
    return df


def build_or_load_vectorstore(client_id: int, format_id: int):
    path = get_vector_path(client_id, format_id)
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=os.getenv("OPENAI_API_KEY"))
    last_created = get_last_created_timestamp(path)
    df = load_audit_data(client_id, format_id, last_created)
    index_file = os.path.join(path, "index.faiss")
    if df.empty:
        if os.path.exists(index_file):
            return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        return None
    if os.path.exists(index_file):
        store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        store.add_texts(
            texts=df["text"].tolist(),
            metadatas=df.drop(columns=["text"]).to_dict(orient="records"),
        )
    else:
        store = FAISS.from_texts(
            texts=df["text"].tolist(),
            metadatas=df.drop(columns=["text"]).to_dict(orient="records"),
            embedding=embeddings,
        )
    os.makedirs(path, exist_ok=True)
    store.save_local(path)
    update_meta(path, str(df["_last_created_max"].iloc[0]))
    return store


def _format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def process_vector_question(question: str, client_id: int, format_id: int) -> dict:
    store = build_or_load_vectorstore(client_id=client_id, format_id=format_id)
    if store is None:
        return {"answer": "", "sources": []}
    retriever = store.as_retriever(search_kwargs={"k": 10})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    prompt = ChatPromptTemplate.from_template(
        "You are a professional data analyst reviewing customer feedback.\n"
        "Analyze the provided records and summarize the key themes and tone.\n"
        "Focus on NPS trends, positive/negative remarks, and branch mentions.\n"
        "- Length: 60–100 words\n"
        "- Style: concise, factual, analytical\n"
        "- Never quote customers directly.\n\n"
        "Question: {question}\n"
        "Feedback Data:\n"
        "{context}"
    )
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    answer = chain.invoke(question).strip()
    top_docs = retriever.invoke(question)
    sources = []
    for doc in top_docs:
        md = getattr(doc, "metadata", {}) or {}
        sources.append(
            {
                "hierarchy_name": md.get("hierarchy_name"),
                "summary": md.get("summary"),
                "nps": md.get("nps"),
                "nps_category": md.get("nps_category"),
                "date_time": md.get("date_time"),
            }
        )
    return {"answer": answer, "sources": sources}
