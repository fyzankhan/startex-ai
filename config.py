import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from openai import OpenAI

load_dotenv()

DB_URL = os.getenv("DB_URL")
if not DB_URL:
    raise ValueError("Missing DB_URL. Please set it in .env")

engine = create_engine(DB_URL, pool_pre_ping=True, echo=False)

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise ValueError("Missing OPENAI_API_KEY. Please set it in .env")

client = OpenAI(api_key=OPENAI_KEY)

print("Config loaded successfully.")
