from pathlib import Path

from llm_model import LLMModel
from logger import get_logger

recipes_logger = get_logger("recipes")


def extract_text(*, image_path: str, output_path: str, model: LLMModel) -> None:
    response_data = model.runtask(image_path)
    recipes_logger.info(response_data)
    with Path(f"{output_path}.txt").open("w") as file:
        file.write(response_data)
