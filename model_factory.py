from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import OpenAIEmbeddings
from common import TaskType, ModelProvider


class ModelFactory:
    def __init__(self, config):
        self.config = config

    def get_model(self, task_type: TaskType) -> BaseChatModel | Embeddings:
        model_configs = self.config.get_model_configs(task_type)
        if not model_configs:
            raise ValueError(f"Unknown task: {task_type}")

        if (
            model_configs.provider == ModelProvider.OPENAI
            and task_type == TaskType.EMBEDDING
        ):
            return OpenAIEmbeddings(openai_api_key=model_configs.key)
        elif model_configs.provider == ModelProvider.OPENAI:
            return ChatOpenAI(
                api_key=model_configs.key,
                model=model_configs.model_name,
                temperature=model_configs.temperature,
            )
        elif model_configs.provider == ModelProvider.GROQ:
            return ChatGroq(
                groq_api_key=model_configs.key,
                model_name=model_configs.model_name,
                temperature=model_configs.temperature,
            )
        else:
            raise ValueError(f"Unknown model type: {model_configs.provider }")
