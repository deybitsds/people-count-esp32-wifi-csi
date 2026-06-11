
from langchain.chat_models import init_chat_model

# LOAD ENV
import os
from config.env import env

api_key_llm = os.getenv("API_KEY_OPEN_ROUTER")
base_url_llm = os.getenv("BASE_URL_OPEN_ROUTER")

from langchain_openai import ChatOpenAI

LLM_USED = ChatOpenAI(
    model='google/gemini-2.5-flash',
    temperature = 0,
    max_retries=8, 
    request_timeout=45,
    api_key=api_key_llm,
    base_url=base_url_llm,
)
