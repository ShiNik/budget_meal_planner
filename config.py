from pydantic.dataclasses import dataclass
from pydantic import ValidationError, Field, constr
from functools import cache
from pathlib import Path
import yaml
from logger import get_logger

recipes_logger = get_logger("recipes")

@dataclass(frozen=True)
class APIKeys:
    openai_api_key: str
    groq_api_key: str

@dataclass(frozen=True)
class DataConfig:
    pdf_path: str
    recipe_books_path: str

    def validate_paths(self) -> None:
        for path in [self.pdf_path, self.recipe_books_path]:
            if not Path(path).exists():
                raise ValueError(f"Path does not exist: {path}")

@dataclass(frozen=True)
class OutputConfig:
    images_path: str
    products_path: str
    recipes_path: str

    def ensure_paths_exist(self) -> None:
        for path in [self.images_path, self.products_path, self.recipes_path]:
            output_path = Path(path)
            if not output_path.exists():
                output_path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:
    api_keys: APIKeys
    data_path: DataConfig
    output_path: OutputConfig
    prompt_files: dict

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
