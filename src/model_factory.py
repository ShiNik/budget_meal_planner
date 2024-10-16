import json

import boto3
from langchain_community.embeddings import BedrockEmbeddings, OpenAIEmbeddings
from langchain_community.llms import Bedrock, SagemakerEndpoint
from langchain_community.llms.sagemaker_endpoint import LLMContentHandler
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from common import ModelProvider, TaskType
from config import BaseModelConfig, Config
from logger import get_logger

recipes_logger = get_logger("recipes")


class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
        input_str = json.dumps({"inputs": prompt, "parameters": model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]["generated_text"]


class ModelFactory:
    def __init__(self, config: Config):
        self.config = config
        self.bedrock = boto3.client(service_name="bedrock-runtime")

    def get_model(self, task_type: TaskType) -> BaseChatModel | Embeddings:
        model_configs = self.config.get_model_configs(task_type)
        recipes_logger.info(f"{task_type}  and model name is {model_configs.model_name}")
        if not model_configs:
            raise ValueError(f"Unknown task: {task_type}")

        if task_type == TaskType.EMBEDDING:
            return self._get_embedding_model(model_configs)

        if task_type in [TaskType.RECOMMEND_RECIPES, TaskType.EXTRACT_PRODUCT]:
            return self._get_llm_model(model_configs)

        raise ValueError(f"Unknown model type: {model_configs.provider }")

    def _get_embedding_model(self, model_configs: BaseModelConfig) -> Embeddings:
        if model_configs.provider == ModelProvider.OPENAI:
            return OpenAIEmbeddings(model=model_configs.model_name, openai_api_key=model_configs.key)

        if model_configs.provider == ModelProvider.BEDROCK_AMAZON:
            return BedrockEmbeddings(model_id=model_configs.model_name, client=self.bedrock)

        if model_configs.provider == ModelProvider.HUGGINGFACE:
            return HuggingFaceEmbeddings(model_name=model_configs.model_name)

        raise ValueError(f"Unknown model type: {model_configs.provider}")

    def _get_llm_model(self, model_configs: BaseModelConfig) -> BaseChatModel:
        if model_configs.provider == ModelProvider.OPENAI:
            return ChatOpenAI(
                api_key=model_configs.key,
                model=model_configs.model_name,
                temperature=model_configs.temperature,
            )

        if model_configs.provider == ModelProvider.GROQ:
            return ChatGroq(
                groq_api_key=model_configs.key,
                model_name=model_configs.model_name,
                temperature=model_configs.temperature,
            )

        if model_configs.provider == ModelProvider.BEDROCK_META:
            return Bedrock(
                model_id=model_configs.model_name,
                client=self.bedrock,
                model_kwargs={"max_gen_len": 512, "temperature": model_configs.temperature},
            )

        if model_configs.provider == ModelProvider.SAGEMAKER:
            client = boto3.client("sagemaker-runtime")
            return SagemakerEndpoint(
                endpoint_name=model_configs.endpoint_name,
                client=client,
                model_kwargs={"temperature": model_configs.temperature},
                content_handler=ContentHandler(),
            )

        raise ValueError(f"Unknown model type: {model_configs.provider}")
