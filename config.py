from pydantic.dataclasses import dataclass
from pydantic import ValidationError, Field
from functools import cache
from pathlib import Path
import yaml
from logger import get_logger

recipes_logger = get_logger("recipes")

@dataclass(frozen=True)
class APIConfigs:
    openai: dict
    groq: dict
    settings: dict

    @property
    def openai_api_key(self) -> str:
        return self.openai['key']

    @property
    def openai_model_name(self) -> str:
        return self.openai['model_name']

    @property
    def openai_url(self) -> str:
        return self.openai['url']

    @property
    def groq_api_key(self) -> str:
        return self.groq['key']

    @property
    def groq_vision_model_name(self) -> str:
        return self.groq['vision_model_name']

    @property
    def groq_model_name(self) -> str:
        return self.groq['model_name']

    @property
    def temperature(self) -> float:
        return self.settings['temperature']

    @property
    def vector_index_path(self) -> str:
        return self.settings['vector_index_path']

@dataclass(frozen=True)
class DataConfig:
    paths: dict

    @property
    def pdf_path(self) -> str:
        return self.paths['pdf']

    @property
    def recipe_books_path(self) -> str:
        return self.paths['recipe_books']

    def validate_paths(self) -> None:
        for path in [self.pdf_path, self.recipe_books_path]:
            if not Path(path).exists():
                raise ValueError(f"Path does not exist: {path}")

@dataclass(frozen=True)
class OutputConfig:
    paths: dict

    @property
    def images_path(self) -> str:
        return self.paths['images']

    @property
    def products_path(self) -> str:
        return self.paths['products']

    @property
    def recipes_path(self) -> str:
        return self.paths['recipes']

    def ensure_paths_exist(self) -> None:
        for path in [self.images_path, self.products_path, self.recipes_path]:
            output_path = Path(path)
            if not output_path.exists():
                output_path.mkdir(parents=True, exist_ok=True)

@dataclass(frozen=True)
class PromptsConfig:
    files: dict

@dataclass(frozen=True)
class Config:
    api_configs: APIConfigs
    data_path: DataConfig
    output_path: OutputConfig
    prompts: PromptsConfig

    def validate_all_paths(self) -> None:
        self.data_path.validate_paths()
        self.output_path.ensure_paths_exist()

@cache
def get_config() -> Config:
    with Path("config.yaml").open() as config_file:
        config_dict = yaml.safe_load(config_file)
        try:
            config = Config(**config_dict)
            config.validate_all_paths()
            return config
        except (ValidationError, ValueError) as e:
            recipes_logger.info("Configuration validation failed:", e)
            raise
