from enum import StrEnum


class ModelProvider(StrEnum):
    OPENAI = "openai"
    GROQ = "groq"
    BEDROCK_AMAZON = "bedrock_amazon"
    BEDROCK_META = "bedrock_meta"
    SAGEMAKER = "sagemaker"


class TaskType(StrEnum):
    EXTRACT_PRODUCT = "extract_product"
    EMBEDDING = "embedding"
    RECOMMEND_RECIPES = "recommend_recipes"
