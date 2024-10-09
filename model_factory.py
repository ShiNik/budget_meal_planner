from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import OpenAIEmbeddings, BedrockEmbeddings
from langchain_community.llms import Bedrock
from common import TaskType, ModelProvider
from config import BaseModelConfig
from langchain_community.llms import SagemakerEndpoint
from langchain_community.llms.sagemaker_endpoint import LLMContentHandler
import boto3
import json

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
    def __init__(self, config):
        self.config = config
        self.bedrock = boto3.client(service_name="bedrock-runtime")

    def get_model(self, task_type: TaskType) -> BaseChatModel | Embeddings:
        model_configs = self.config.get_model_configs(task_type)
        print(task_type, model_configs.model_name,)
        if not model_configs:
            raise ValueError(f"Unknown task: {task_type}")

        if task_type == TaskType.EMBEDDING:
            return self._get_embedding_model(model_configs)
        elif task_type in [TaskType.RECOMMEND_RECIPES, TaskType.EXTRACT_PRODUCT]:
            return self._get_llm_model(model_configs)
        else:
            raise ValueError(f"Unknown model type: {model_configs.provider }")

    def _get_embedding_model(self, model_configs: BaseModelConfig):
        if model_configs.provider == ModelProvider.OPENAI:
            return OpenAIEmbeddings(openai_api_key=model_configs.key)
        elif model_configs.provider == ModelProvider.BEDROCK_AMAZON:

            return BedrockEmbeddings(model_id=model_configs.model_name, client=self.bedrock)
        else:
            raise ValueError(f"Unknown model type: {model_configs.provider}")

    def _get_llm_model(self, model_configs: BaseModelConfig):
        if model_configs.provider == ModelProvider.OPENAI:
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
        elif model_configs.provider == ModelProvider.BEDROCK_META:
            return Bedrock(model_id=model_configs.model_name,
                          client=self.bedrock,
                          model_kwargs={'max_gen_len': 512, "temperature":model_configs.temperature})
        elif model_configs.provider == ModelProvider.SAGEMAKER:
            client = boto3.client("sagemaker-runtime")
            SagemakerEndpoint(
                endpoint_name=model_configs.endpoint_name,
                client=client,
                model_kwargs={"temperature": model_configs.temperature},
                content_handler=ContentHandler(),
            )

        else:
            raise ValueError(f"Unknown model type: {model_configs.provider}")
