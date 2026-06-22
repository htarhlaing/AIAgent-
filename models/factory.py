import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


load_dotenv()


def require_google_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 GOOGLE_API_KEY，请复制 .env.example 为 .env 并填写密钥。")
    return api_key


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Embeddings | BaseChatModel:
        pass

class ChatModelFactory(BaseModelFactory):
    def generator(self) -> BaseChatModel:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=require_google_api_key(),
        )


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Embeddings:
        return GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=require_google_api_key(),
        )


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
