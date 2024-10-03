from enum import StrEnum, auto

class ModelProvider(StrEnum):
    OPENAI = "openai"
    GROQ = "groq"

class TaskType(StrEnum):
    EXTRACT_PRODUCT = "extract_product"
    EMBEDDING = "embedding"
    RECOMMEND_RECIPES = "recommend_recipes"